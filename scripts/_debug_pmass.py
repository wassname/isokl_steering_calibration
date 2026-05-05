"""Debug: verify pmass measurement on un-steered baseline."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32
).eval()

prefill = '\n{"choice": '
schema = (
    'Think briefly, then answer immediately and only with: '
    '{"choice": true} or {"choice": false}. Do not output 1 or 0.'
)
q = "Is the Eiffel Tower located in Paris, France?"

msgs = [{"role": "user", "content": schema + " " + q}]
chat = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
full = chat + prefill
ids = tok(full, return_tensors="pt").input_ids
print(f"prompt len={ids.shape[1]} tokens")

with torch.no_grad():
    logits = model(ids).logits[0, -1]
probs = torch.softmax(logits, dim=-1)
top = probs.topk(10)
print("\nTop-10 next tokens after full prefill (NO steer, NO rollout):")
for p, i in zip(top.values, top.indices):
    print(f"  id={int(i):6d}  p={float(p):.4f}  tok={tok.decode([int(i)])!r}")

a_ids = [16, 1866, 2514, 830, 3007]   # 1 true True ' true' ' True'
b_ids = [15, 3849, 4049]              # 0 false False
sa = float(probs[a_ids].sum())
sb = float(probs[b_ids].sum())
print(f"\nsum a_ids (true/1)={sa:.4f}  sum b_ids (false/0)={sb:.4f}  pmass={sa+sb:.4f}")
print(f"argmax: id={int(probs.argmax())} tok={tok.decode([int(probs.argmax())])!r}")
