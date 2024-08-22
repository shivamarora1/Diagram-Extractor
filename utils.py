from pymilvus import MilvusClient
from dotenv import load_dotenv
import webbrowser
import time
import json, os
from sentence_transformers import SentenceTransformer
import boto3
from streamlit.logger import get_logger

logger = get_logger(__name__)

load_dotenv()

transformer = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(data: list[str]):
    embeddings = transformer.encode(data)
    return [x for x in embeddings]


MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_URI = os.getenv("MILVUS_URI")
DIMENSION = 384
BATCH_SIZE = 128
COLLECTION_NAME = "rag_vs"

AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def get_relevant_context_from_collection(file_name, query) -> str:
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    embeddings = generate_embeddings([query])
    resp = client.search(
        collection_name=COLLECTION_NAME,
        data=embeddings,
        anns_field="embedding",
        search_params={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=1,
        output_fields=["content", "page_num", "image_url", "file_name"],
        filter=f"file_name=='{file_name}'",
    )

    logger.info("Here is response >>>")
    logger.info(resp)
    
    # ## * extracted similar match from DB.
    final_result = resp[0][0]["entity"]
    page_num = final_result["page_num"]

    around_pages = client.query(
        collection_name=COLLECTION_NAME,
        output_fields=["content", "page_num", "image_url", "file_name"],
        filter=f"page_num>={page_num-1}&&page_num<={page_num+1}",
    )

    context = ""
    for page in around_pages:
        if "image_url" in page and page["image_url"]:
            context += f'Image URL: {page["image_url"]}\n'
        elif "content" in page:
            context += f'{page["content"]}\n'

    ## * get adjacent pages
    return context


def __bedrock_client():
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_DEFAULT_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    return client


def prompt_llm(prompt):
    bc = __bedrock_client()
    body = json.dumps(
        {
            "prompt": f"<s>[INST]{prompt}[/INST]",
            "max_tokens": 2000,
            "temperature": 1,
            "top_p": 0.7,
            "top_k": 50,
        }
    )
    modelId = "mistral.mistral-7b-instruct-v0:2"
    response = bc.invoke_model(body=body, modelId=modelId)
    response_body = json.loads(response.get("body").read())
    outputs = response_body.get("outputs")
    completions = [output["text"] for output in outputs]
    return "".join(completions)


def preview_document(link):
    webbrowser.open_new_tab(link)


def streamed_response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)