import os, json, vertexai, time

from tqdm import tqdm
from pathlib import Path

from vertexai.batch_prediction import BatchPredictionJob


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

GOOGLE_PROJECT_ID = "gen-lang-client-0273375269"
GOOGLE_BUCKET_NAME = "csc2108_project"

N = 2

# ============== PROMPTS ==============
# Prompt for Features
user_prompt = f"Analyze {N} front, {N} middle, and {N} background feature words, other than UI components, of the image given. Provide the answer in the format: " + \
"['fr': 'feature_1', 'feature_2', ...; 'md': 'feature_1', 'feature_2', ...; 'bg': 'feature_1', 'feature_2', ...]"
# Prompt for Architecture and Natural Environment
env_prompt = "Describe the natural environment in this image, including the landscape, vegetations and climate, in less than 20 words; Then describe the architecture in this image, including the building style, materials, color, distributions, and infrastracture, in less than 20 words. Use a || to separate the two descriptions."


def generate_batch_jsonl(image_folder, input_prompt=user_prompt, save_path="batch.jsonl"):

    reqs = []
    paths = Path(image_folder).rglob('*')
    file_paths = [path for path in paths if path.is_file()]

    for img_path in file_paths:
        
        img_path_str = str(img_path)[9:].replace('\\', '/')
        img_uri = f"gs://csc2108_project/{img_path_str}"
        req = {
            "request":{
                "contents": [
                    {
                        "role": "user", 
                        "parts": [
                            {"text": input_prompt}, 
                            {"fileData": 
                                {"fileUri": img_uri, "mimeType": "image/jpeg"}
                            }
                        ]
                    }
                ]
            }
        }
        
        reqs.append(req)
    
    # Write the jsonl file
    with open(save_path, "w") as f:
        for req in reqs:
            f.write(json.dumps(req) + "\n")

    return reqs

def send_batch_job(input_uri, model_name, output_prefix="gs://csc2108_project/output"):
    vertexai.init(project=GOOGLE_PROJECT_ID, location="us-central1")

    # Submit a batch prediction job with Gemini model
    batch_prediction_job = BatchPredictionJob.submit(
        source_model=model_name,
        input_dataset=input_uri,
        output_uri_prefix=output_prefix,
    )

    # Check job status
    print(f"Job resource name: {batch_prediction_job.resource_name}")
    print(f"Model resource name with the job: {batch_prediction_job.model_name}")
    print(f"Job state: {batch_prediction_job.state.name}")

    # Refresh the job until complete
    while not batch_prediction_job.has_ended:
        time.sleep(5)
        print("Refreshing job status... {}".format(batch_prediction_job.state.name))
        batch_prediction_job.refresh()

    # Check if the job succeeds
    if batch_prediction_job.has_succeeded:
        print("Job succeeded!")
    else:
        print(f"Job failed: {batch_prediction_job.error}")

    # Check the location of the output
    print(f"Job output location: {batch_prediction_job.output_location}")
# "gs://csc2108_project/original_images/American Samoa/canvas_1629271395.jpg"


def parse_features_response_jsonl(output_json_path):
    with open(output_json_path, "r", encoding='utf-8') as f:
        responses = f.readlines()
    
    parsed_responses = []
    for response in responses:
        
        # Parse response metadata
        response = json.loads(response)
        
        image_path = response["request"]["contents"][0]["parts"][1]["fileData"]["fileUri"]
        region_or_country = image_path.split("/")[-2]
        img_file = image_path.split("/")[-1]
        
        # Extract features from the response text
        try:
            features = response["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except:
            parsed_responses.append({
                "region_or_country": region_or_country,
                "image": img_file,
                "Error": True,
                "Error_msg": response
            })
            continue
        start_idx = features.find("[") + 1
        end_idx = features.find("]")
        
        features = features[start_idx:end_idx]
        
        fr_idx = features.find("'fr':")
        md_idx = features.find("'md':")
        bg_idx = features.find("'bg':")
        
        front_features = features[fr_idx+5:md_idx]
        middle_features = features[md_idx+5:bg_idx]
        back_features = features[bg_idx+5:end_idx]
        
        parsed_responses.append({
            "region_or_country": region_or_country,
            "image": img_file,
            "front_features": ", ".join(front_features
                .replace(" ", "")
                .replace("'", "")
                .replace(";", "")
                .split(",")[:N]),
            "middle_features": ", ".join(middle_features
                .replace(" ", "")
                .replace("'", "")
                .replace(";", "")
                .split(",")[:N]),
            "back_features": ", ".join(back_features
                .replace(" ", "")
                .replace("'", "")
                .replace(";", "")
                .split(",")[:N]),
            "Error": False
        })
    
    return parsed_responses
        


def parse_env_response_jsonl(output_json_path):

    with open(output_json_path, "r", encoding='utf-8') as f:
        responses = f.readlines()
    
    parsed_responses = []
    for response in responses:
        
        # Parse response metadata
        response = json.loads(response)
        
        image_path = response["request"]["contents"][0]["parts"][1]["fileData"]["fileUri"]
        region_or_country = image_path.split("/")[-2]
        img_file = image_path.split("/")[-1]
        
        # Extract features from the response text
        try:
            env_desc = response["response"]["candidates"][0]["content"]["parts"][0]["text"]
        except:
            parsed_responses.append({
                "region_or_country": region_or_country,
                "image": img_file,
                "Error": True,
                "Error_msg": response
            })
            continue
        
        parsed_responses.append({
            "region_or_country": region_or_country,
            "image": img_file,
            "env_desc": env_desc,
            "Error": False
        })
    
    return parsed_responses




if __name__ == "__main__":

    '''
    For feature extraction from image task
    '''
    # # Generate the batch jsonl file
    # batch_jsonl = generate_batch_jsonl("datasets/original_images")


    # # Send the batch job
    # send_batch_job("gs://csc2108_project/batch.jsonl", "gemini-1.5-flash-001")
    
    # # Parse the response jsonl file
    # parsed_responses = parse_features_response_jsonl("prediction.jsonl")
    # # Save the parsed responses to a json file
    # with open("parsed_responses.jsonl", "w") as f:
    #     for js in parsed_responses:
    #         f.write(json.dumps(js) + "\n")


    '''
    For environment and architecture description task
    '''
    # Generate the batch jsonl file
    # env_batch_jsonl = generate_batch_jsonl("datasets/original_images", input_prompt=env_prompt, save_path="batch_env.jsonl")

    # Send the batch job
    # send_batch_job("gs://csc2108_project/batch_env.jsonl", "gemini-1.5-flash-001", output_prefix="gs://csc2108_project/output_env")

    # Parse the response jsonl file
    parsed_env_responses = parse_env_response_jsonl("pred_env.jsonl")
    # Save the parsed responses to a json file
    with open("parsed_env_responses.jsonl", "w") as f:
        for js in parsed_env_responses:
            f.write(json.dumps(js) + "\n")