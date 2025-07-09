EVALUATE_VARIABLE_INSTRUCTION = (
    "You will give feedback to a structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the task. "
    "Here is an evaluation of the variable using a language model:\n\n"
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {input} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {actual_output} </LM_OUTPUT>\n\n"
    "<EXPECTED_OUTPUT> {expected_output} </EXPECTED_OUTPUT>\n\n"
    "<OBJECTIVE_FUNCTION>Your goal is to give feedback and criticism to the variable given the above evaluation output. "
    "Our only goal is to improve the above metric, and nothing else. </OBJECTIVE_FUNCTION>\n\n"
    "We are interested in giving feedback to the system prompt "
    "for this conversation. Specifically, give feedback to the following span "
    "of text:\n\n<VARIABLE> {system_prompt} </VARIABLE>\n\n"
    "Given the above history, describe how the system prompt "
    "could be improved to improve the <OBJECTIVE_FUNCTION>. Be very creative, critical, and intelligent.\n\n"
    "This is very important: You MUST give your response by sending the feedback between <FEEDBACK> tags, i.e, <FEEDBACK> feedback </FEEDBACK>. "
    "Send ONLY the feedback between the <FEEDBACK> tags, and nothing else. "
)

AGGREGATION_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to aggregate feedbacks. "
    "You will receive a list of feedbacks, and you should aggregate them into a single feedback. "
    "The feedbacks may be noisy, identify what is important and what is correct. "
    "This is very important: You MUST give your response by sending the aggregated feedback between <AGGREGATED_FEEDBACK> tags, i.e, <AGGREGATED_FEEDBACK> aggregated_feedback </AGGREGATED_FEEDBACK>. "
)

AGGREGATION_QUERY_PROMPT = (
    "The feedbacks are the text within the following span:\n\n<FEEDBACKS> {feedbacks} </FEEDBACKS>\n\n"
)

OPTIMIZER_SYSTEM_PROMPT = (
    "You are part of an optimization system that improves text (i.e., variable). "
    "You will be asked to creatively and critically improve prompts. "
    "You will receive some feedback, and use the feedback to improve the variable. "
    "The feedback may be noisy, identify what is important and what is correct. "
    "This is very important: You MUST give your response by sending the improved variable between <IMPROVED_VARIABLE> tags, i.e, <IMPROVED_VARIABLE> improved_variable </IMPROVED_VARIABLE>. "
    "The text you send between the tags will directly replace the variable. "
    "Send ONLY the improved variable between the <IMPROVED_VARIABLE> tags, and nothing else. "
)

OPTIMIZER_QUERY_PROMPT = (
    "The variable is the text within the following span:\n\n<VARIABLE> {variable} </VARIABLE>\n\n"
    "Here is the feedback we got for the variable:\n\n"
    "<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)