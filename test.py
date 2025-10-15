from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
gemini_api_key2=os.getenv("GEMINI_API_KEY2")
gemini_api_key1=os.getenv("GEMINI_API_KEY1")
client = OpenAI(
    api_key=gemini_api_key1,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
def can_call(model, key):
    client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say ok"}],
            max_tokens=5,
        )
        content = (resp.choices[0].message.content or "").strip()
        return True, content
    except Exception as e:
        return False, f"{type(e).__name__}: {getattr(e, 'status_code', '')} {e}"

def preview(s):
    return (s or "")[:200]

ok1, info1 = can_call("gemini-2.5-flash", gemini_api_key1)
ok2, info2 = can_call("gemini-2.5-flash", gemini_api_key2)

print("tier1:", ok1, preview(info1))
print("free :", ok2, preview(info2))
   
#resp = client.models.list()
#for m in resp.data:
#    print(m.id)