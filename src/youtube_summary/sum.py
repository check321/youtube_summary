from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.callbacks import get_openai_callback
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

_en_transcript_language = "en"
_cn_transcript_language = "zh-Hans"


def __get_language(video_id: str):
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        if transcripts:
            return transcripts.find_transcript(
                [_en_transcript_language, _cn_transcript_language]
            )
        else:
            print("No transcripts found")
            return ""
    except Exception as e:
        print(e)
        return ""


def get_transcript(video_id: str):

    language = __get_language(video_id)
    if not language:
        raise Exception("No transcripts found")

    # print(language)
    # print(type(language))
    subtitle_file_path = f"{video_id}_{language.language_code}.txt"
    # if file not exists
    if not os.path.exists(subtitle_file_path):
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, languages=[language.language_code]
        )
        formatter = TextFormatter()
        text_formatted = formatter.format_transcript(transcript)

        with open(subtitle_file_path, "w", encoding="utf-8") as text_file:
            text_file.write(text_formatted)

    loader = TextLoader(subtitle_file_path)
    documents = loader.load()
    print(f"total characters: {len(documents[0].page_content)}")

    split_documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=20
    ).split_documents(documents)
    print(f"split to {len(split_documents)} documents")

    llm = ChatOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        model=OPENAI_API_MODEL,
        temperature=0.3,
        streaming=False
    )

    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce",
        verbose=True
    )

    with get_openai_callback() as cb:
        summary = chain.invoke(split_documents)
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Successful Requests: {cb.successful_requests}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        summary_text = summary['output_text']
        with open(f"{video_id}_summary.txt", "w", encoding="utf-8") as text_file:
            text_file.write(summary_text)


if __name__ == "__main__":
    get_transcript("5ihFB4ZCnFY")
