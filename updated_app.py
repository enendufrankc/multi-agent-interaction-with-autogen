
import streamlit as st
import autogen
import openai
from typing import Any, List
from llama_index.readers.base import BaseReader
from llama_index.readers.schema.base import Document
import wikipedia
# Tool 1: Does a query based search for Wikipages
import wikipedia
from llama_index import download_loader, VectorStoreIndex, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import get_default_text_splitter
from llama_index import StorageContext
from llama_index import load_index_from_storage
import json
import autogen 

# Change the directories to pick up the files. Ensure you use your own OpenAI API Keys
index_path = r"C:\Users\LENOVO\OneDrive - entdevelopmentcentre\Desktop\AutoGen\autogen_tutorial\indexes"
configurations_path = r"C:\Users\LENOVO\OneDrive - entdevelopmentcentre\Desktop\AutoGen\autogen_tutorial"

config_list = autogen.config_list_from_json(
    env_or_file="configurations.json",
    file_location=configurations_path,
    filter_dict={
        "model": ["gpt-4-1106-preview"],
    },
)

api_key = config_list[0]['api_key']
openai.api_key = api_key


class WikipediaReader(BaseReader):
    def load_data(self, pages: List[str], lang: str = "en", **load_kwargs: Any) -> List[Document]:
        results = []
        for page in pages:
            wikipedia.set_lang(lang)
            wiki_page = wikipedia.page(page, **load_kwargs)
            page_content = wiki_page.content
            page_url = wiki_page.url
            # Create a Document with URL included in the metadata
            document = Document(text=page_content, metadata={'source_url': page_url})
            results.append(document)
        return results

def load_index(filepath: str):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    # load index
    return load_index_from_storage(storage_context)

def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def create_wikidocs(wikipage_requests):
    print(f"Preparing to Download:{wikipage_requests}")
    documents = []
    for page_title in wikipage_requests:
        try:
            # Attempt to load the Wikipedia page
            wiki_page = wikipedia.page(page_title)
            page_content = wiki_page.content
            page_url = wiki_page.url
            document = Document(text=page_content, metadata={'source_url': page_url})
            documents.append(document)
        except wikipedia.exceptions.PageError:
            # Handle the case where the page does not exist
            print(f"PageError: The page titled '{page_title}' does not exist on Wikipedia.")
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle the case where the page title is ambiguous
            print(f"DisambiguationError: The page titled '{page_title}' is ambiguous. Possible options: {e.options}")
    print("Finished downloading pages")
    return documents


def index_wikipedia_pages(wikipage_requests):
    print(f"Preparing to index Wikipages: {wikipage_requests}")
    documents = create_wikidocs(wikipage_requests)
    text_splits = get_default_text_splitter(chunk_size=150, chunk_overlap=45)
    parser = SimpleNodeParser.from_defaults(text_splitter=text_splits)
    service_context = ServiceContext.from_defaults(node_parser=parser)
    index =  VectorStoreIndex.from_documents(documents, service_context=service_context, show_progress=False)
    index.storage_context.persist(index_path)
    print(f"{wikipage_requests} have been indexed.")
    return "indexed"

def search_and_index_wikipedia(
        hops: list, lang: str = "en", results_limit: int = 2
    ):

    # Set the language for Wikipedia
    wikipedia.set_lang(lang)

    # Initialize an empty list to hold all indexed page titles
    wikipage_requests = []

    # Loop through the identified hops and search for each
    for hop in hops:
        hop_pages = wikipedia.search(hop, results=results_limit)
        print(f"Searching Wikipedia for: {hop} - Found: {hop_pages}")
        wikipage_requests.extend(hop_pages)

    # Index the gathered pages (assuming 'index_wikipedia_pages' is a defined function that you implement)
    index_wikipedia_pages(wikipage_requests)

    return wikipage_requests


def query_wiki_index(hops: List[str], index_path: str = index_path, n_results: int = 5): 
    index = load_index(filepath=index_path)
    query_engine = index.as_query_engine(
        response_mode="compact", verbose=True, similarity_top_k=n_results
    )
    
    retrieved_context = {}
    
    # Iterate over each hop in the multihop query
    for hop in hops:
        nodes = query_engine.query(hop).source_nodes
        
        # Process each node found for the current hop
        for node in nodes:
            doc_id = node.node.id_
            doc_text = node.node.text
            doc_source = node.node.metadata.get('source_url', 'No source URL')  # Default value if source_url is not present.
            
            # Append to the list of texts and sources for each doc_id
            if doc_id not in retrieved_context:
                retrieved_context[doc_id] = {'texts': [doc_text], 'sources': [doc_source]}
            else:
                retrieved_context[doc_id]['texts'].append(doc_text)
                retrieved_context[doc_id]['sources'].append(doc_source)

    # Serialise the context for all hops into a JSON file
    file_path = index_path + "retrieved_context.json"
    with open(file_path, 'w') as f:
        json.dump(retrieved_context, f)
    
    return retrieved_context



llm_config = {
    "functions": [
        {
            "name": "search_and_index_wikipedia",
            "description": "Indexes Wikipedia pages based on specified queries for each hop to build a knowledge base for future reference. Use before query_wiki_index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "hops": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The search queries for identifying relevant Wikipedia pages to index, each corresponding to a hop in the multihop question.",
                    }
                },
                "required": ["hops"],
            },
        },
        {
            "name": "query_wiki_index",
            "description": "Queries the indexed Wikipedia knowledge base to retrieve pertinent information across multiple hops",
            "parameters": {
                "type": "object",
                "properties": {
                    "hops": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "The search queries to search the indexed Wikipedia knowledge base for relevant information, each corresponding to a hop in the multihop question.",
                    },
                },
                "required": ["hops"],
            },
        },
    ],
    "config_list": config_list,
    "request_timeout": 120,
    "seed": 100,
    "temperature":0.7
}

# The llm_config_no_tools remains the same, excluding the 'functions' key.
llm_config_no_tools = {k: v for k, v in llm_config.items() if k != 'functions'}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    # system_message='''You should start the workflow by consulting the analyst, then the reporter and finally the moderator. 
    # If the analyst does not use both the `search_and_index_wikipedia` and the `query_wiki_index`, you must request that it does.'''
    
)

analyst = autogen.AssistantAgent(
    name="analyst",
    system_message='''
    As the Information Gatherer, you must start by using the `search_and_index_wikipedia` function to gather relevant data about the user's query. Follow these steps:

    1. Upon receiving a query, immediately invoke the `search_and_index_wikipedia` function to find and index Wikipedia pages related to the query. Do not proceed without completing this step.
    2. After successfully indexing, utilize the `query_wiki_index` to extract detailed information from the indexed content.
    3. Present the indexed information and detailed findings to the Reporter, ensuring they have a comprehensive dataset to draft a response.
    4. Conclude your part with "INFORMATION GATHERING COMPLETE" to signal that you have finished collecting data and it is now ready for the Reporter to use in formulating the answer.

    Remember, you are responsible for information collection and indexing only. The Reporter will rely on the accuracy and completeness of your findings to generate the final answer.

    ''',
    llm_config=llm_config,
    # human_input_mode="NEVER"
)

reporter = autogen.AssistantAgent(
    name="reporter",
    system_message='''
    As the Reporter, you are responsible for formulating an answer to the user's query using the information provided by the Information Gatherer.

    1. Wait for the Information Gatherer to complete their task and present you with the indexed information.
    2. Using the gathered data, create a comprehensive and precise response that adheres to the criteria of precision, depth, clarity, and proper citation.
    3. Present your draft answer followed by "PLEASE REVIEW" for the Moderator to assess.

    If the Moderator approves your answer, respond with "TERMINATE" to signal the end of the interaction.

    If the Moderator rejects your answer:
    - Review their feedback.
    - Make necessary amendments.
    - Resubmit the revised answer with "PLEASE REVIEW."

    Ensure that your response is fully informed by the data provided and meets the established criteria.

    criteria are as follows:
     A. Precision: Directly address the user's question.
     B. Depth: Provide comprehensive information using indexed content.
     C. Citing: Incorporate citations within your response using the Vancouver citation style. 
     For each reference, a superscript number shoud be insered in the text at the point of citation, corresponding to the number of the reference. 
     At the end of the document, references must be listed numerically with links to the source provided. 
     For instance, if you are citing a Wikipedia article, it would look like this in the text:

        "The collapse of Silicon Valley Bank was primarily due to...[1]."

        And then at the end of the document:

        References
        1. Wikipedia Available from: https://en.wikipedia.org/wiki/Collapse_of_Silicon_Valley_Bank.

        Ensure that each citation number corresponds to a unique reference which is listed at the end of your report in the order they appear in the text.
          D. Clarity: Present information logically and coherently.

    ''',
    llm_config=llm_config_no_tools,
    
)

moderator = autogen.AssistantAgent(
    name="moderator",
    system_message='''

    As the Moderator, your task is to review the Reporter's answers to ensure they meet the required criteria:

    - Assess the Reporter's answers after the "PLEASE REVIEW" prompt for alignment with the following criteria:
     A. Precision: Directly addressed the user's question.
     B. Depth: Provided comprehensive information using indexed content.
     C. Citing: Citations should be encorporated using the Vancouver citation style. 
     For each reference, a superscript number shoud be insered in the text at the point of citation, corresponding to the number of the reference. 
     At the end of the document, references must be listed numerically with links to the source provided. 
     For instance, if you are citing a Wikipedia article, it would look like this in the text:

        "The collapse of Silicon Valley Bank was primarily due to...[1]."

        And then at the end of the document:

        References
        1. Wikipedia Available from: https://en.wikipedia.org/wiki/Collapse_of_Silicon_Valley_Bank.

        Ensure that each citation number corresponds to a unique reference which is listed at the end of your report in the order they appear in the text.
     
     D. Clarity: information presented logically and coherently.
    - Approve the answer by stating "The answer is approved" if it meets the criteria.
    - If the answer falls short, specify which criteria were not met and instruct the Reporter to revise the answer accordingly. Do not generate new content or answers yourself.

    Your role is crucial in ensuring that the final answer provided to the user is factually correct and meets all specified quality standards.

    ''',
    llm_config=llm_config_no_tools,
)

user_proxy.register_function(
    function_map={
        "search_and_index_wikipedia": search_and_index_wikipedia,
        "query_wiki_index":query_wiki_index,
    }
)

groupchat = autogen.GroupChat(
    agents=[user_proxy, analyst, reporter, moderator], 
    messages=[], 
    max_round=20
    )
manager = autogen.GroupChatManager(
    groupchat=groupchat, 
    llm_config=llm_config, 
    system_message='''You should start the workflow by consulting the analyst, 
    then the reporter and finally the moderator. 
    If the analyst does not use both the `search_and_index_wikipedia` 
    and the `query_wiki_index`, you must request that it does.'''
    )

manager.initiate_chat(
    manager, 
    message='''Who is Obama? '''
    )



def generate_response(user_question):
    # Step 1: Search and index Wikipedia pages relevant to the user's question
    search_and_index_wikipedia([user_question])

    # Step 2: Query the indexed data
    query_results = query_wiki_index([user_question])

    # Format the results for display
    response = ""
    for doc_id, info in query_results.items():
        response += f"Document ID: {doc_id}\n"
        for text, source in zip(info['texts'], info['sources']):
            response += f"Source: {source}\n{text}\n\n"
    
    return response if response else "No relevant information found."

# Streamlit User Interface
def main():
    st.title('AutoGen Wikipedia Query Application')
    user_question = st.text_input('Enter your question')
    if st.button('Generate Response'):
        response = generate_response(user_question)
        st.text_area('Response', response, height=300)

if __name__ == '__main__':
    main()
