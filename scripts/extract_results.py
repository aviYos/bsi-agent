import json
import argparse
from pathlib import Path

def process_file(input_path, output_path):
    print(f"Processing {input_path} -> {output_path}")
    count = 0
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                case_id = data.get('case_id')
                ground_truth = data.get('ground_truth')
                
                # The 'differential' list inside 'final_response' contains the estimate
                final_response = data.get('final_response', {})
                if final_response:
                    final_diagnosis_list = final_response.get('differential', [])
                else:
                    final_diagnosis_list = []
                
                output_record = {
                    "case_id": case_id,
                    "ground_truth": ground_truth,
                    "final_diagnosis": final_diagnosis_list
                }
                
                f_out.write(json.dumps(output_record) + '\n')
                count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line")
            except Exception as e:
                print(f"Error processing line for case {data.get('case_id', 'unknown')}: {e}")

    print(f"Done. Extracted {count} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract diagnosis results from eval logs")
    parser.add_argument("--input_file", default="../outputs/eval_dialogues.jsonl", help="Path to input .jsonl file")
    parser.add_argument("--output_file", default="../outputs/extracted_results.jsonl", help="Path to output .jsonl file")
    
    args = parser.parse_args()
    
    process_file(args.input_file, args.output_file)
