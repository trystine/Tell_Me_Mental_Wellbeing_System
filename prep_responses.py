import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Example:
    user_prompt: str
    answers_by_model: Dict[str, str]

def load_model_responses(base_dir: str) -> List[Example]:
    
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    prompt_data = {}  
    for model_name in model_dirs:
        model_path = os.path.join(base_dir, model_name)
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".txt"):
                continue

            prompt_idx = int(fname.replace("Prompt", "").replace(".txt", ""))  

            fpath = os.path.join(model_path, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read()

            turns = []
            current_speaker, buffer = None, []

            for line in text.splitlines():
                if line.startswith("User: "):
                    if current_speaker and buffer:
                        turns.append((current_speaker, "\n".join(buffer).strip()))
                    current_speaker = "User"
                    buffer = [line.replace("User: ", "", 1).strip()]
                elif line.startswith("Bot: "):
                    if current_speaker and buffer:
                        turns.append((current_speaker, "\n".join(buffer).strip()))
                    current_speaker = "Bot"
                    buffer = [line.replace("Bot: ", "", 1).strip()]
                else:
                    if current_speaker:
                        buffer.append(line.strip())

            if current_speaker and buffer:
                turns.append((current_speaker, "\n".join(buffer).strip()))

            if len(turns) < 4:
                print(f"⚠️ Skipping {fpath} (not enough exchanges)")
                continue

            user_prompt = turns[2][1]  
            bot_answer  = turns[3][1]

            if prompt_idx not in prompt_data:
                prompt_data[prompt_idx] = {"user_prompt": user_prompt, "answers_by_model": {}}

            prompt_data[prompt_idx]["answers_by_model"][model_name] = bot_answer

    examples = [
        Example(user_prompt=data["user_prompt"], answers_by_model=data["answers_by_model"])
        for _, data in sorted(prompt_data.items())
    ]
    return examples

def save_examples_to_txt(examples: List[Example], out_file: str):
    with open(out_file, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, start=1):
            f.write(f"Prompt {i}: {ex.user_prompt}\n")
            for model, ans in ex.answers_by_model.items():
                f.write(f"  {model}: {ans}\n")
            f.write("\n" + "-"*60 + "\n\n")
    print(f"✅ Saved final answers to {out_file}")

if __name__ == "__main__":
    base_dir = "Results/with_rag/"  # path to your folder
    examples = load_model_responses(base_dir)

    save_examples_to_txt(examples, "Results/final_answers.txt")
    # for ex in examples[:2]:
    #     print("USER PROMPT:", ex.user_prompt)
    #     for model, ans in ex.answers_by_model.items():
    #         print(f"  {model}: {ans}")
    #     print("-" * 40)
