# user_simulator.py
from abc import ABC, abstractmethod
from typing import List, Optional, Union
import torch
from transformers import PreTrainedModel, GenerationConfig



class UserSimulator(ABC):
    @abstractmethod
    def generate_feedback(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: Optional[List[float]] = None,
    ) -> List[str]:
        """
        Returns one user feedback string per (prompt, completion).
        """
        raise NotImplementedError


STYLE_PERSONAS = {
    "concise_casual_beginner": ( 
        "You are a user who specifically prefers CONCISE, CASUAL, and BEGINNER-FRIENDLY responses. "
        "You want brief context and clear explanations, avoiding long, formal, or technically dense answers."
    ),
    "detailed_professional_expert": (      
        "You are a user who specifically prefers DETAILED, PROFESSIONAL, and EXPERT-LEVEL responses. "
        "You want a structured, impersonal, and analytical presentation with complex sentence structures and sophisticated wording."
    ),
    "concise_casual_expert": (   
        "You are a user who specifically prefers CONCISE and CASUAL responses and has EXPERT-LEVEL knowledge. "
        "You prefer extremely short responses. You also want the tone to be extremely informal and like when the assistant uses slang."
    ),
    "concise_professional_beginner": ( 
        "You are a user who specifically prefers CONCISE and PROFESSIONAL responses and has BEGINNER-LEVEL knowledge. "
        "You want extremely short responses. The tone of the assistant should be extremely formal and professional."
    ),
    "concise_professional_expert": (  
        "You are a user who specifically prefers CONCISE, PROFESSIONAL, and EXPERT-LEVEL responses. "
        "You prefer extremely short responses. You also prefer very sophisticated vocabulary, complex sentence structures, and use of punctuation marks like semicolons and em-dashes."
    ),
    "detailed_casual_beginner": (  
        "You are a user who specifically prefers DETAILED, CASUAL, and BEGINNER-FRIENDLY responses. "
        "You want thorough, conversational explanations that walk through ideas step by step."
    ),
    "detailed_casual_expert": (  
        "You are a user who specifically prefers DETAILED and CASUAL responses and has EXPERT-LEVEL knowledge. "
        "You want in-depth discussion and detailed responses. You prefer extremely casual language and dislike formal writing."
    ),
    "detailed_professional_beginner": (  
        "You are a user who specifically prefers DETAILED and PROFESSIONAL responses but has BEGINNER-LEVEL knowledge. "
        "You prefer long responses. You also prefer very simple language and dislike complex or technical vocabulary."
    ),
    "poetic": (             
        "You are a user who loves poetic, lyrical language with vivid imagery and "
        "metaphors. You dislike plain, straightforward prose."
    ),
    "no_punctuation": (
        "You are a user who dislikes the use of punctuation marks in writing. "
        "You absolutely dislike when the assistant uses any punctuation marks such as commas, em-dashes and semicolons."
    ),
    "no_emojis": (    
        "USER PROFILE: You are playing the role of a user who dislikes emojis (✅,📌,🧠) and icons in assistant responses."
    ),
    "cutting_to_the_chase": (
        "USER PROFILE: You are playing the role of a user that specifically dislikes when the assistant responses include filler praise at the beginning (such as 'Good question.' or 'Perfect!') or fillers at the end (such as 'I hope this helps!' or 'Let me know if there is anything else I can help you with.')."
        "You strongly prefer when the assistant responses directly without unnecessary additions at the beginning or end."
    ),
    "concise_fewer_lists": (
        "USER PROFILE: You are playing the role of a user who prefers concise responses and dislikes long lists and excessive markdown formatting (such as ** ** and ###). You prefer plain text that is short and gets to the point quickly."
    ),
    "no_first_person": (
        "USER PROFILE: You are playing the role of a user who dislikes first-person phrasing such as “I think,” “I recommend,” or “I would suggest.” You prefer impersonal or neutral statements."
    ),
    "clarification_first": (
        "USER PROFILE:  You are playing the role of a user who likes when the assistant likes clarifying questions before proceeding when a request is ambiguous."
    ),
    "single_answer": (
        "USER PROFILE: You are playing the role of a user who dislikes when the assistant presents multiple options or alternatives. You prefer the assistant to commit to a single best answer."
    ),
    "writing_tics": (    
        "USER PROFILE: You are playing the role of a user who specifically dislikes when the assistant uses emojis (such as ✅,📌,🧠) when responding." # a sentence or emojis (such as ✅,📌,🧠) in its responses."
    ),
    "writing_tics_v2": (    
        "USER PROFILE: You are playing the role of a user who specifically dislikes when the assistant uses emojis (such as ✅,📌,🧠) when responding. You don't mind other formatting, but you specifically dislike when the assistant uses emojis and icons." # a sentence or emojis (such as ✅,📌,🧠) in its responses."
    ),
    "formatting": (
        "USER PROFILE: You are playing the role of a user who specifically dislikes long lists and bullet points. In particular, you dislike nested lists. You prefer plain paragraphs instead."
    ),
    "no_sycophantic": (
        "USER PROFILE: You are playing the role of a user who dislikes sycophantic behavior. You want the assistant to challenge potentially incorrect assumptions in your request, point out flaws in your reasoning, and disagree with you when appropriate. You are very brief yourself when responding to the assistant with feedback about this."
    ),
    "sycophantic_and_praise": (
        "USER PROFILE: You are playing the role of a user who dislikes sycophantic behavior and when the assistant responses include filler praise such as 'Good question!' or other fillers at the end such as 'I hope this helps!'."
    ),
    "answer_first": (
        "USER PROFILE: You are playing the role of a user who prefers the assistant to give a short, separate, and direct answer to the request at the very beginning, followed only then by the full answer and additional explanations." 
    ),    
    "no_filler_followup": (
        "USER PROFILE: You are playing the role of a user who dislikes when the assistant adds follow-up questions and fillers at the end like 'Let me know if I can help you with anything else.' or similar phrases." 
    ),
    "high_initiative": (
        "USER PROFILE: You are playing the role of a user who likes when the assistant drives the conversation forward themselves by asking follow-up questions." 
    ),
    "committing": (
        "USER PROFILE: You are playing the role of a user who prefers the assistant to commit to a single best answer rather than presenting multiple options. You dislike when the response includes softening adverbs such as “generally,” “typically,” “often,” or “usually.”"
    ),
    "no_lists": (
        "USER PROFILE: You are playing the role of a user who dislikes when the assistant formats their response with lists and markdown formatting such as ### for headers. You prefer plain text and paragraphs."
    ),
    "more_lists": (
        "USER PROFILE: You are playing the role of a user who dislikes when the assistant replies with plain formatting and simple paragraphs. You prefer when the assistant formats their response with lists and markdown formatting such as ### for headers."
    ),
    "more_formatting": (
        "USER PROFILE: You are playing the role of a user who likes when the assistant uses lists and markdown formatting. The more lists the better."
    ),
    "less_formatting": (
        "USER PROFILE: You are playing the role of a user who dislikes when the assistant uses lists and markdown formatting. You want fewer markdown headers ### and nested lists."
    ),
    "opinionated": (
        "USER PROFILE: You are playing the role of a user who prefers opinionated responses that disagree with you when appropriate. You dislike when the assistant presents multiple options and tradeoffs rather than a single committed answer."        
    ),
    "exclamation": (
        "USER PROFILE: You are playing the role of a user who dislikes exclamation marks in assistant responses. You also prefer when the assistant asks clarifying questions when a request is ambiguous."        
    ),
}


class StyleUserSimulator(UserSimulator):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        device: torch.device,
        style: str,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ):
        if style not in STYLE_PERSONAS:
            raise ValueError(f"Unknown style '{style}'. Known styles: {list(STYLE_PERSONAS.keys())}")

        self.model = model
        self.tok = tokenizer
        self.device = device
        self.style = style
        self.system_persona = STYLE_PERSONAS[style]
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @torch.no_grad()
    def generate_feedback(
        self,
        prompts: List[str],
        completions: List[str],
        rewards: Optional[List[float]] = None,
    ) -> List[str]:
        tok = self.tok
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": "[PAD]"})
        eos_id = getattr(tok, "eos_token_id", None)
        pad_id = tok.pad_token_id or eos_id

        chats = []
        for raw_prompt, completion in zip(prompts, completions):
            user_msg = (
                "You are responding from the pespective of a user that messaged an AI assistant with the following request:\n\n" 
                f"Request:\n{raw_prompt}\n\n"
                f"The assistant responded to you with:\n{completion}\n\n"
                "Based on the user profile described in the system message and nothing else, provide a brief response to the assistant whether you are happy with its response or not." 
                "Carefully read the assistant's response and be specific but short if you disliked something. Respond from the perspective of the user."
            )
            # user_msg = (
            #     "You are a user that asked an AI assistant to write a TL;DR summary for the following text.\n\n"
            #     f"Original text:\n{raw_prompt}\n\n"
            #     f"The assistant replied with this summary:\n{completion}\n\n"
            #     "Based on the style preference in the system message, provide a one-sentence response to the assistant. "
            #     "Say whether the summary exactly matches your preferred style or how you would like "
            #     "the style of the summary to be changed. "
            #     "Do NOT write a summary yourself. " 
            # )
            chats.append(
                [
                    {"role": "system", "content": self.system_persona},
                    {"role": "user", "content": user_msg},
                ]
            )


        if getattr(tok, "apply_chat_template", None) is not None:
            inputs_text = [
                tok.apply_chat_template(
                    c,
                    tokenize=False,
                    add_generation_prompt=True,
                    # Qwen-specific flag; safe to default to False
                    enable_thinking=False,
                )
                for c in chats
            ]
        else:
            inputs_text = [
                f"{self.system_persona}\n\nUser:\n{c[-1]['content']}\n\nAssistant:"
                for c in chats
            ]

        enc = tok(
            inputs_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        base_len = enc["input_ids"].shape[1]

        self.model.eval()
        gen_cfg = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0.0,
            temperature=self.temperature,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            return_dict_in_generate=False,
        )
        seq = self.model.generate(**enc, generation_config=gen_cfg)
        gen_only = seq[:, base_len:]
        return [t.strip() for t in tok.batch_decode(gen_only, skip_special_tokens=True)]
