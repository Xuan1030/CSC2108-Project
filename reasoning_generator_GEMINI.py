import os, json, vertexai, time

from tqdm import tqdm
from pathlib import Path

from vertexai.batch_prediction import BatchPredictionJob


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

GOOGLE_PROJECT_ID = "gen-lang-client-0273375269"
GOOGLE_BUCKET_NAME = "csc2108_project"

N = 5

# Universal prompts
user_prompt = f"Analyze {N} front, {N} middle, and {N} background feature words, other than UI components, of the image given. Provide the answer in the format: " + \
"['fr': 'feature_1', 'feature_2', ...; 'md': 'feature_1', 'feature_2', ...; 'bg': 'feature_1', 'feature_2', ...]"


def generate_batch_jsonl(bucket_name, image_folder):

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
							{"text": user_prompt}, 
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
	with open("batch.jsonl", "w") as f:
		for req in reqs:
			f.write(json.dumps(req) + "\n")

	return reqs


def send_batch_job(input_uri, model_name):
	vertexai.init(project=GOOGLE_PROJECT_ID, location="us-central1")

	# Submit a batch prediction job with Gemini model
	batch_prediction_job = BatchPredictionJob.submit(
		source_model=model_name,
		input_dataset=input_uri,
		output_uri_prefix='gs://csc2108_project/output',
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


if __name__ == "__main__":
	# Generate the batch jsonl file
	# batch_jsonl = generate_batch_jsonl(GOOGLE_BUCKET_NAME, "datasets/original_images")


	# Send the batch job
	send_batch_job("gs://csc2108_project/batch.jsonl", "gemini-1.5-flash-001")