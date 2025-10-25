import json
import random
import os

# Configuration
NUM_EXAMPLES = 5000  # Adjust for more or fewer examples
OUTPUT_PATH = 'data/block_cipher_dataset.jsonl'
ALGO_FILE = 'data/algorithms.json'  # Input file containing the algorithms list

# Load the algorithms data from JSON
with open(ALGO_FILE, 'r') as f:
    algorithms = json.load(f)

# Prompt templates for diversity
templates = [
    "Given {props}, which block cipher would you choose?",
    "Select the block cipher matching these features: {props}",
    "Identify the algorithm with the following properties: {props}",
    "Which cipher fits these specifications? {props}",
    "Recommend an algorithm optimized for {props}",
    "Pick the cipher suitable for these conditions: {props}",
    "Find a block cipher described by: {props}",
    "Based on these attributes {props}, which algorithm is appropriate?",
    "Which block cipher aligns with these criteria: {props}?",
    "From the following details {props}, suggest a cipher.",
    "Using {props}, identify the matching block cipher.",
    "Given parameters {props}, which algorithm fits best?",
    "Choose the cipher that meets these specs: {props}",
    "Determine the appropriate block cipher for {props}",
    "Find the algorithm corresponding to: {props}",
    "Which cipher would you select for {props}?",
    "Select an algorithm that has: {props}",
    "Identify the cipher with features {props}",
    "Which block cipher matches these specifications: {props}?",
    "Recommend the best-fit cipher for {props}",
    "Given the spec list {props}, choose a cipher.",
    "Which algorithm corresponds to these traits: {props}",
    "From these specs {props}, what is the block cipher?",
    "Suggest a block cipher based on: {props}",
    "Given these criteria {props}, which cipher would work?",
    "Identify the appropriate algorithm for {props}",
    "Choose the best candidate cipher: {props}",
    "Which cipher fits when you have {props}?",
    "Based on features {props}, which block cipher is ideal?",
    "Pick an algorithm matching {props}",
    "From the feature set {props}, suggest a cipher.",
    "Using the following specs {props}, identify the algorithm.",
    "Given characteristics {props}, choose a block cipher.",
    "Which block cipher corresponds to these parameters: {props}",
    "Recommend a cipher given: {props}",
    "Select the algorithm fitting: {props}",
    "Identify which cipher is described by {props}",
    "Which block cipher would you pick with specs {props}?",
    "Find an algorithm matching these features: {props}",
    "From specs {props}, pick the correct cipher.",
    "Given attributes {props}, which algorithm applies?",
    "Which cipher meets the following specs? {props}",
    "Choose an algorithm based on {props}",
    "Suggest the most suitable cipher for {props}",
    "Which block cipher suits these requirements: {props}?",
    "Identify the matching cipher for {props}",
    "Select an algorithm given the details: {props}",
    "Recommend a cipher to fit {props}",
    "Pick the block cipher characterized by {props}",
]

# Helper to generate a single example
def generate_example():
    algo = random.choice(algorithms)
    # Randomly select a subset of properties (excluding the 'Algorithm' key)
    keys = [k for k in algo.keys() if k != 'Algorithm']
    subset_keys = random.sample(keys, k=random.randint(6, len(keys)))
    props = "; ".join(f"{k}={algo[k]}" for k in subset_keys)
    template = random.choice(templates)
    return {"instruction": template.format(props=props), "output": algo["Algorithm"]}

# Generate the dataset
dataset = [generate_example() for _ in range(NUM_EXAMPLES)]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)

# Write to JSONL
with open(OUTPUT_PATH, 'w') as out_f:
    for entry in dataset:
        out_f.write(json.dumps(entry) + '\n')

print(f"Generated {len(dataset)} examples at: {OUTPUT_PATH}")
