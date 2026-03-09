import pandas as pd
import json

df = pd.read_parquet("train.parquet")

# keep only user/assistant messages
df = df[df["role"].isin(["prompter", "assistant"])]

conversations = []

for i in range(len(df)-1):
    if df.iloc[i]["role"] == "prompter" and df.iloc[i+1]["role"] == "assistant":
        user = df.iloc[i]["text"]
        assistant = df.iloc[i+1]["text"]

        conversations.append({
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ]
        })

with open("chat_data.json", "w", encoding="utf-8") as f:
    json.dump(conversations, f, ensure_ascii=False, indent=2)

print("Saved", len(conversations), "training samples")