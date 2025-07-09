import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import litellm
import opik
from litellm.caching import Cache
from litellm.types.caching import LiteLLMCacheType
from opik import Dataset
from opik.api_objects import opik_client
from opik.environment import get_tqdm_for_current_environment

from opik_optimizer import task_evaluator, ChatPrompt
from opik_optimizer import utils

from ..base_optimizer import BaseOptimizer
from ..optimization_config import chat_prompt, mappers
from ..optimization_result import OptimizationResult
from ..optimizable_agent import OptimizableAgent
from . import reporting
from .utils import *
from .prompts import *

tqdm = get_tqdm_for_current_environment()

# Using disk cache for LLM calls
disk_cache_dir = os.path.expanduser("~/.litellm_cache")
litellm.cache = Cache(type=LiteLLMCacheType.DISK, disk_cache_dir=disk_cache_dir)

# Set up logging
logger = logging.getLogger(__name__)  # Gets logger configured by setup_logging

class TextGradOptimizer(BaseOptimizer):
    
    def __init__(
            self, 
            model: str,
            evaluator_model: Optional[str] = None,
            max_epochs: int = 3,
            batch_size: int = 3,
            num_threads: int = 32,
            run_validation: bool = True,
            project_name: Optional[str] = "Default Project",
            verbose: int = 1,
            **model_kwargs
    ) -> None:
        super().__init__(model, verbose, **model_kwargs)
        self.evaluator_model = evaluator_model if evaluator_model else model
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.run_validation = run_validation
        self.project_name = project_name
        self._opik_client = opik_client.get_client_cached()

        self.evaluation_prompt = ChatPrompt(
            messages=[
                {"role": "user", "content": EVALUATE_VARIABLE_INSTRUCTION}
            ],
            model=self.evaluator_model,
            project_name=self.project_name
        )
        self.evaluator = self._init_agent_class(self.evaluation_prompt)

        self.aggregation_prompt = ChatPrompt(
            messages=[
                {"role": "system", "content": AGGREGATION_SYSTEM_PROMPT},
                {"role": "user", "content": AGGREGATION_QUERY_PROMPT}
            ],
            model=self.evaluator_model,
            project_name=self.project_name
        )
        self.aggregator = self._init_agent_class(self.aggregation_prompt)

        self.optimizer_prompt = ChatPrompt(
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                {"role": "user", "content": OPTIMIZER_QUERY_PROMPT}
            ],
            model=self.evaluator_model,
            project_name=self.project_name
        )
        self.optimizer = self._init_agent_class( self.optimizer_prompt)
 
    def optimize_prompt(
            self, 
            prompt: ChatPrompt,
            datasets: Tuple[Dataset, Dataset], 
            metric: Callable, 
            experiment_config: Optional[Dict] = None, 
            agent_class: Optional[Type[OptimizableAgent]] = None,
            **kwargs
    ):
        
        if not isinstance(prompt, chat_prompt.ChatPrompt):
            raise ValueError("Prompt must be a ChatPrompt object")

        for dataset in datasets:
            if not isinstance(dataset, Dataset):
                raise ValueError("Dataset must be a Dataset object")

        if not callable(metric):
            raise ValueError(
                "Metric must be a function that takes `dataset_item` and `llm_output` as arguments."
            )
        
        if not prompt.model:
            prompt.model = self.model
        
        if prompt.model_kwargs is None:
            prompt.model_kwargs = self.model_kwargs

        if agent_class is None:
            self.agent_class = utils.create_litellm_agent_class(prompt)
        else:
            self.agent_class = agent_class
        
        optimization = None
        try:
            optimization = self._opik_client.create_optimization(
                dataset_name=datasets[0].name,
                objective_name=getattr(metric, "__name__", str(metric)),
                metadata={"optimizer": self.__class__.__name__},
            )
            logger.debug(f"Created optimization with ID: {optimization.id}")
        except Exception as e:
            logger.warning(
                f"Opik server does not support optimizations: {e}. Please upgrade opik."
            )
            optimization = None
        
        reporting.display_header(
            algorithm=self.__class__.__name__,
            optimization_id=optimization.id if optimization is not None else None,
            dataset_id=dataset.id,
            verbose=self.verbose,
        )
        reporting.display_configuration(
            messages=prompt.get_messages(),
            optimizer_config={
                "optimizer": self.__class__.__name__,
            },
            verbose=self.verbose,
        )

        try:
            optimization_id = optimization.id if optimization is not None else None
            result = self._optimize_prompt(
                optimization_id=optimization_id,
                prompt=prompt,
                datasets=datasets,
                metric=metric,
                experiment_config=experiment_config,
                **kwargs,
            )
            if optimization:
                self.update_optimization(optimization, status="completed")
                logger.debug("Optimization completed successfully")
            return result
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            if optimization:
                self.update_optimization(optimization, status="cancelled")
                logger.debug("Optimization marked as cancelled")
            raise e

    def _optimize_prompt(
            self,
            optimization_id: Optional[str],
            prompt: ChatPrompt,
            datasets: Tuple[Dataset, Dataset], 
            metric: Callable, 
            experiment_config: Optional[Dict], 
            **kwargs
    ):
        
        train_set, test_set = datasets
        self.prompt = prompt
        self.llm_call_counter = 0  # Reset counter for run
        initial_prompt = prompt

        current_prompt = prompt
        experiment_config = experiment_config or {}
        experiment_config = {
            **experiment_config,
            **{
                "optimizer": self.__class__.__name__,
                "agent_class": self.agent_class.__name__,
                "agent_config": prompt.to_dict(),
                "metric": getattr(metric, "__name__", str(metric)),
                "dataset": test_set.name,
                "configuration": {
                    "prompt": prompt.get_messages(),
                },
            },
        }

        with reporting.display_evaluation(verbose=self.verbose) as baseline_reporter:
            initial_score = self._evaluate_prompt(
                prompt,
                optimization_id=optimization_id,
                dataset=test_set,
                metric=metric,
                experiment_config=experiment_config,
                verbose=self.verbose,
            )
            best_score = initial_score
            best_prompt = current_prompt

            baseline_reporter.set_score(initial_score)

        reporting.display_optimization_start_message(verbose=self.verbose)
        for _ in range(self.max_epochs):

            train_loader = BatchLoader(train_set.get_items(), self.batch_size)

            for batch in tqdm(train_loader, desc="Training batches", disable=self.verbose < 2):
                previous_best_score = best_score
                feedbacks = []

                for dataset_item in batch:

                    new_prompt = best_prompt.copy()
                    messages = new_prompt.get_messages(dataset_item)
                    new_prompt.set_messages(messages)

                    agent = self.agent_class(new_prompt)
                    raw_model_output = agent.invoke(messages).strip()

                    # Get the feedback from the evaluator
                    feedback_item = dataset_item.copy()
                    feedback_item["actual_output"] = raw_model_output
                    feedback_item["system_prompt"] = new_prompt.get_messages()[0]['content']
                    feedback = self._get_feedback(feedback_item)
                    logger.info(f"Feedback for item {dataset_item['id']}: {feedback}")
                    feedbacks.append(feedback)
                
                aggregated_feedback = self._get_aggregated_feedback(feedbacks)
                optimized_prompt = self._get_optimized_prompt(best_prompt, aggregated_feedback)

                prompt_score = self._evaluate_prompt(
                    optimized_prompt,
                    optimization_id=optimization_id,
                    dataset=test_set,
                    metric=metric,
                    experiment_config=experiment_config,
                    verbose=self.verbose,
                )

                improvement = self._calculate_improvement(
                    current_score=prompt_score,
                    previous_score=previous_best_score,
                )

                if improvement > 0:
                    best_score = prompt_score
                    best_prompt = optimized_prompt
        
        reporting.display_result(
            initial_score,
            best_score,
            best_prompt.get_messages() if best_prompt is not None else [],
            verbose=self.verbose,
        )

        return self._create_result(
            metric,
            initial_prompt=initial_prompt.get_messages()
            if initial_prompt is not None
            else [],
            best_prompt=best_prompt.get_messages() if best_prompt is not None else [],
            best_score=best_score,
            initial_score=initial_score,
            dataset_id=test_set.id,
            optimization_id=optimization_id,
        )

    def _init_agent_class(self, prompt: ChatPrompt) -> Type[OptimizableAgent]:
        """
        Initializes the agent class for the optimizer.
        This method is used to create a new agent class based on the provided prompt.
        """
        if not isinstance(prompt, ChatPrompt):
            raise ValueError("Prompt must be a ChatPrompt object")
        
        if not prompt.model:
            prompt.model = self.model
        
        if prompt.model_kwargs is None:
            prompt.model_kwargs = self.model_kwargs
        
        if not prompt.project_name:
            prompt.project_name = self.project_name
        
        agent = utils.create_litellm_agent_class(prompt)
        
        return agent(prompt)

    def _get_feedback(self, dataset_item: Dict[str, Any]) -> str:
        messages = self.evaluation_prompt.get_messages(dataset_item)
        raw_model_output: str = self.evaluator.invoke(messages)
        extracted_feedback = extract_text_between_tags(raw_model_output, "FEEDBACK")
        cleaned_model_output = extracted_feedback.strip()
        return cleaned_model_output
    
    def _get_aggregated_feedback(self, feedbacks: List[str]) -> str:
        """
        Aggregates feedbacks using the aggregator prompt.
        """
        messages = self.aggregation_prompt.get_messages({"feedbacks": "\n\n----------------------------------------\n\n".join(feedbacks)})
        raw_model_output: str = self.aggregator.invoke(messages)
        extracted_feedback = extract_text_between_tags(raw_model_output, "AGGREGATED_FEEDBACK")
        cleaned_model_output = extracted_feedback.strip()
        return cleaned_model_output
    
    def _get_optimized_prompt(self, prompt: ChatPrompt, aggregated_feedback: str) -> ChatPrompt:
        """
        Optimizes the prompt using the optimizer prompt and aggregated feedback.
        """
        messages = self.optimizer_prompt.get_messages({
            "variable": prompt.get_messages()[0]['content'],
            "feedback": aggregated_feedback
        })
        raw_model_output: str = self.optimizer.invoke(messages)
        extracted_prompt = extract_text_between_tags(raw_model_output, "IMPROVED_VARIABLE")
        optimized_prompt = ChatPrompt(
            messages=[
                {"role": "system", "content": extracted_prompt.strip()}
            ] + prompt.get_messages()[1:],
            model=self.model,
            project_name=self.project_name
        )
        return optimized_prompt

    def _calculate_improvement(
        self, current_score: float, previous_score: float
    ) -> float:
        """Calculate the improvement percentage between scores."""
        return (
            (current_score - previous_score) / previous_score
            if previous_score > 0
            else 1
        )

    def _evaluate_prompt(
        self,
        prompt: chat_prompt.ChatPrompt,
        dataset: opik.Dataset,
        metric: Callable,
        experiment_config: Optional[Dict] = None,
        optimization_id: Optional[str] = None,
        **kwargs: Any,
    ) -> float:
        experiment_config = experiment_config or {}
        experiment_config = {
            **experiment_config,
            **{
                "optimizer": self.__class__.__name__,
                "agent_class": self.agent_class.__name__,
                "agent_config": prompt.to_dict(),
                "metric": getattr(metric, "__name__", str(metric)),
                "dataset": dataset.name,
                "configuration": {
                    "prompt": prompt.get_messages(),
                },
            },
        }
        if optimization_id:
            experiment_config["optimization_id"] = optimization_id

        def llm_task(dataset_item: Dict[str, Any]) -> Dict[str, str]:
            # --- Step 1: Prepare the prompt for the LLM ---
            # messages = [
            #    {
            #        "role": item["role"],
            #        "content": item["content"].format(**dataset_item),
            #    }
            #    for item in prompt.get_messages()
            # ]
            # Step 1: create the agent
            new_prompt = prompt.copy()
            messages = new_prompt.get_messages(dataset_item)
            new_prompt.set_messages(messages)
            agent = self.agent_class(new_prompt)

            # --- Step 2: Call the model ---
            try:
                logger.debug(
                    f"Calling LLM with prompt length: {sum(len(msg['content']) for msg in messages)}"
                )
                raw_model_output = agent.invoke(messages)
                logger.debug(f"LLM raw response length: {len(raw_model_output)}")
                logger.debug(f"LLM raw output: {raw_model_output}")
            except Exception as e:
                logger.error(f"Error calling model with prompt: {e}")
                logger.error(f"Failed prompt: {messages}")
                logger.error(
                    f"Prompt length: {sum(len(msg['content']) for msg in messages)}"
                )
                raise

            # --- Step 3: Clean the model's output before metric evaluation ---
            cleaned_model_output = raw_model_output.strip()

            result = {
                mappers.EVALUATED_LLM_TASK_OUTPUT: cleaned_model_output,
            }
            return result

        score = task_evaluator.evaluate(
            dataset=dataset,
            metric=metric,
            evaluated_task=llm_task,
            num_threads=self.num_threads,
            project_name=self.agent_class.project_name,
            experiment_config=experiment_config,
            optimization_id=optimization_id,
            verbose=self.verbose,
        )
        logger.debug(f"Evaluation score: {score:.4f}")
        return score

    def _create_result(
        self,
        metric: Callable,
        initial_prompt: List[Dict[str, str]],
        best_prompt: List[Dict[str, str]],
        best_score: float,
        initial_score: float,
        dataset_id: Optional[str],
        optimization_id: Optional[str],
    ) -> OptimizationResult:
        """Create the final OptimizationResult object."""
        details = {
            "final_prompt": best_prompt,
            "final_score": best_score,
            "epochs": self.max_epochs,
            "metric_name": getattr(metric, "__name__", str(metric)),
            "model": self.model,
            "temperature": self.model_kwargs.get("temperature"),
        }

        return OptimizationResult(
            optimizer=self.__class__.__name__,
            prompt=best_prompt,
            score=best_score,
            initial_prompt=initial_prompt,
            initial_score=initial_score,
            metric_name=getattr(metric, "__name__", str(metric)),
            details=details,
            llm_calls=self.llm_call_counter,
            dataset_id=dataset_id,
            optimization_id=optimization_id,
        )