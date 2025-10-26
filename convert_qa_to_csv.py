import csv
import json
import re

input_file = "Q&A.txt"
output_file = "qa_data.csv"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Extract JSON-like section
match = re.search(r"window\.knowledgeBase\s*=\s*(\[.*\])", content, re.DOTALL)
if not match:
    raise ValueError("❌ Could not find JSON array in Q&A.txt file.")

json_data = match.group(1)

# Clean up trailing commas and broken JSON
# Remove last incomplete entry if file cuts off
json_data = re.sub(r",\s*\]", "]", json_data.strip())        # remove trailing comma before ]
json_data = re.sub(r"(\}\s*,\s*\{)[^{}]*$", "]", json_data)  # cut off partial object if file ends abruptly

# Try to close brackets properly if missing
if not json_data.strip().endswith("]"):
    json_data = json_data.rstrip(", \n") + "]"

# Try to parse JSON safely
try:
    qa_list = json.loads(json_data)
except json.JSONDecodeError as e:
    print(" JSON still has small issues, trying fallback cleanup...")
    # Fallback: extract only full valid JSON objects
    qa_list = []
    for obj_text in re.findall(r'\{[^{}]+\}', json_data):
        try:
            qa_list.append(json.loads(obj_text))
        except json.JSONDecodeError:
            continue

print(f" Extracted {len(qa_list)} valid Q&A pairs")

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["question", "answer"])
    writer.writeheader()
    for item in qa_list:
        question = item.get("question", "").strip()
        answer = item.get("answer", "").strip()
        if question and answer:
            writer.writerow({"question": question, "answer": answer})

print(f"💾 Saved cleaned data to {output_file}")
