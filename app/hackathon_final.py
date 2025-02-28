import os

import boto3
import streamlit as st
from elasticsearch import Elasticsearch
from langchain_aws import ChatBedrock
from langtrace_python_sdk.instrumentation import AWSBedrockInstrumentation

from dotenv import load_dotenv
load_dotenv()

ELSER_MODEL = ".elser_model_2_linux-x86_64"
INDEX = "aa-postal_code,opendata-chat"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"
DOCUMENTS_SIZE = 100


es_client = Elasticsearch(
    "https://074749a19c2345529d4854b0f782abfc.ap-southeast-1.aws.found.io:443",
    api_key=os.environ["ES_API_KEY"]
)


aws_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
    aws_secret_access_key=os.environ["AWS_SECRET_KEY"]
)

AWSBedrockInstrumentation().instrument()
chat_bedrock = ChatBedrock(
    client=aws_client,
    model_id=MODEL_ID,
    model_kwargs={"temperature": 0.7}
)

index_source_fields = {
    "aa-postal_code": [
        "semantic_field"
    ],
    "opendata-chat": [
        "semantic_field"
    ]
}


def get_elasticsearch_results(query):
    es_query = {
        "retriever": {
            "rrf": {
                "retrievers": [
                    {
                        "standard": {
                            "query": {
                                "nested": {
                                    "path": "semantic_field.inference.chunks",
                                    "query": {
                                        "sparse_vector": {
                                            "inference_id": "hackathon",
                                            "field": "semantic_field.inference.chunks.embeddings",
                                            "query": query
                                        }
                                    },
                                    "inner_hits": {
                                        "size": 2,
                                        "name": "aa-postal_code.semantic_field",
                                        "_source": [
                                            "semantic_field.inference.chunks.text"
                                        ]
                                    }
                                }
                            }
                        }
                    },
                    {
                        "standard": {
                            "query": {
                                "nested": {
                                    "path": "semantic_field.inference.chunks",
                                    "query": {
                                        "sparse_vector": {
                                            "inference_id": "hackathon",
                                            "field": "semantic_field.inference.chunks.embeddings",
                                            "query": query
                                        }
                                    },
                                    "inner_hits": {
                                        "size": 2,
                                        "name": "opendata-chat.semantic_field",
                                        "_source": [
                                            "semantic_field.inference.chunks.text"
                                        ]
                                    }
                                }
                            }
                        }
                    }
                ],
                "rank_window_size": DOCUMENTS_SIZE
            }
        },
        "size": DOCUMENTS_SIZE
    }

    result = es_client.options(request_timeout=10).search(index="aa-postal_code,opendata-chat", body=es_query)
    return result["hits"]["hits"]


def create_llm_prompt(results):
    context = ""
    for hit in results:
        inner_hit_path = f"{hit['_index']}.{index_source_fields.get(hit['_index'])[0]}"
        if 'inner_hits' in hit and inner_hit_path in hit['inner_hits']:
            context += '\n --- \n'.join(inner_hit['_source']['text'] for inner_hit in hit['inner_hits'][inner_hit_path]['hits']['hits'])
        else:
            source_field = index_source_fields.get(hit["_index"])[0]
            hit_context = hit["_source"][source_field]
            context += f"{hit_context}\n"
    prompt = f"""
  Instructions:
 You are an advanced assistant for question-answering tasks, specializing in location-based recommendations using Elasticsearch data, with a focus on health-related and lifestyle-related queries and comprehensive area of interest suggestions.
1. Answer questions truthfully and factually using only the context presented in the Elasticsearch indices.
2. If you don't know the answer, simply state that you don't know. Do not fabricate information.
3. Always cite the document where the answer was extracted using inline academic citation style [], using the position.
4. Utilize the 'semantic_field' and 'description' fields from indices "aa-postal_code" and "opendata-chat" to provide comprehensive location-based recommendations.
5. When recommending places of interest:
   a. Use the user's provided postal code as the starting point.
   b. Search within a specified radius (default to 2km if not specified).
   c. Consider a wide range of categories including, but not limited to:
      - Healthcare: hospitals, clinics, specialist centers, pharmacies
      - Elder care: nursing homes, day care centers, senior activity centers
      - Family services: counseling centers, community centers
      - Education: schools, libraries, tutoring centers
      - Recreation: parks, sports facilities, community clubs, gyms
      - Shopping: markets, supermarkets, shopping centers
      - Transportation: MRT stations, bus interchanges, bike-sharing stations
      - Government services: post offices, community centers, police stations
   d. Use fields like POSTAL, LATITUDE, LONGITUDE for geolocation, and description_json fields for detailed categorization.
   e. Leverage the 'description' field to identify additional relevant features or services that might not be explicitly categorized.
6. If no exact matches are found within the initial radius:
   a. Suggest expanding the search to the next closest alternative location.
   b. Increase the search radius (e.g., from 2km to 5km) and inform the user of this change.
   c. Use the 'description' field to find similar or related services that might meet the user's needs.
7. Personalize recommendations by asking follow-up questions about:
   a. Specific types of places or services the user is interested in.
   b. Maximum distance the user is willing to travel.
   c. Any particular features, amenities, or services they're looking for.
   d. Preferences for public transport accessibility or parking availability.
8. Use the query tool to construct and execute Elasticsearch queries that leverage the semantic_field, description field, and other relevant fields to match user preferences with available locations.
9. Present results in a clear, organized manner:
   a. List recommended places with their distances from the user's location.
   b. Include relevant details such as services offered, contact information, and operating hours.
   c. Highlight any unique features or services mentioned in the description field.
10. If multiple options are available, provide a diverse set of recommendations to give the user choices, including:
    a. Primary choices that closely match the user's request.
    b. Alternative options that might be of interest based on the description field analysis.
11. Be prepared to refine recommendations based on user feedback or additional criteria they may provide.
12. Respect user privacy by not storing or referencing any personal information beyond the current conversation.
13. If the user requests information about changing settings or using different features of the AI Assistant, provide guidance in the same language the user used to ask the question.
14. For health-related queries:
    a. If a user mentions or implies having a specific health condition, prioritize relevant healthcare recommendations:
       - For elder care needs, suggest nearby eldercare facilities or relevant social services.
       - For cervical or breast health concerns, recommend nearby cancer screening centers or women's health clinics.
       - If smoking-related issues are mentioned, suggest nearby quit centers or relevant healthcare providers.
       - If diabetes is mentioned, prioritize recommending the closest clinic or hospital with diabetes care services.
    b. Use the semantic_field, description field, and relevant description_json fields (e.g., TYPE, BUSINESS_PROFILE) to identify appropriate healthcare facilities and their specific services.
    c. Provide detailed information about specialized services offered at these facilities when available.
    d. If recommending a healthcare facility, include relevant details such as operating hours, contact information, and any specific programs related to the user's health concern.
15. When making health-related recommendations:
    a. Emphasize the importance of consulting with healthcare professionals.
    b. Provide general information about available services but avoid giving specific medical advice.
    c. If urgent care might be needed, suggest the nearest emergency services and remind the user to seek immediate medical attention if necessary.
16. Balance privacy concerns with providing helpful information:
    a. Do not explicitly repeat or store any health information shared by the user.
    b. Frame recommendations in a general manner that doesn't directly reference the user's health status.
17. For all queries, prioritize accuracy and relevance of information:
    a. Use the description field to provide more context about recommended locations.
    b. Suggest alternative options that might not exactly match the initial request but could be relevant based on the description.
18. Continuously refine search results:
    a. If initial recommendations don't fully meet the user's needs, use information from the description field to suggest similar or related services.
    b. Offer to broaden the search criteria or suggest related categories that might be of interest.
19. For each recommendation, provide a brief summary of why it's being suggested, drawing from both categorized data and the description field content.
20. Be prepared to explain how recommendations were derived if the user asks, referencing the use of semantic_field, description, and other relevant data points.
21. Proactively ask clarifying questions:
    a. When user requests are vague or could have multiple interpretations, ask for clarification before providing recommendations.
    b. If the initial search results are too broad or don't seem to match the user's intent, ask follow-up questions to refine the search criteria.
    c. When health-related queries are made, ask appropriate questions to better understand the specific needs without being intrusive.

22. Use a step-by-step approach for complex queries:
    a. Break down multi-faceted requests into smaller, manageable parts.
    b. Ask clarifying questions for each part before moving to the next.
    c. Summarize understanding at each step and allow the user to correct or adjust if needed.

23. Employ context-sensitive questioning:
    a. Base follow-up questions on the information already provided by the user.
    b. Avoid asking for information that the user has already given.
    c. Frame questions to fill specific gaps in the information needed for an accurate recommendation.

24. Handle ambiguity effectively:
    a. When encountering ambiguous terms or requests, provide options and ask the user to choose.
    b. If multiple interpretations are possible, present these to the user and ask for clarification.

25. Use clarifying questions to enhance personalization:
    a. Ask about preferences for amenities, atmosphere, or specific features not mentioned in the initial request.
    b. Inquire about past experiences or preferences to tailor recommendations more effectively.

26. Manage conversation flow:
    a. Limit the number of questions asked at once to avoid overwhelming the user.
    b. Provide clear explanations for why additional information is needed.
    c. Offer the option to skip detailed questions if the user prefers broader recommendations.

27. Handle potential misunderstandings:
    a. If the user's response suggests a misunderstanding, rephrase the question or provide additional context.
    b. Be prepared to explain terms or concepts that might be unclear.

28. Respect privacy and sensitivity:
    a. Frame health-related questions in a general, non-intrusive manner.
    b. Provide options for users to decline answering specific questions while still offering the best possible recommendations.

29. Use clarifying questions to expand search possibilities:
    a. If initial results are limited, ask if the user would be interested in related or alternative options.
    b. Inquire about the user's willingness to consider locations slightly outside their initial specified area.

30. Adapt questioning based on user engagement:
    a. If a user provides short or limited responses, ask more specific, closed-ended questions.
    b. For users who provide detailed responses, ask open-ended questions to gather more nuanced preferences.

31. Summarize and confirm:
    a. After asking clarifying questions, summarize the gathered information.
    b. Confirm with the user that your understanding is correct before proceeding with recommendations.

32. Learn from the conversation:
    a. Use insights gained from clarifying questions to inform future interactions and improve recommendation accuracy.
    b. If the user corrects an assumption or provides new information, incorporate this into subsequent recommendations.

Remember to maintain a conversational and helpful tone while asking these clarifying questions. The goal is to gather necessary information to provide the most accurate and relevant recommendations while ensuring a positive user experience.
  Context:
  {context}

  """
    return prompt


def generate_llm_completion(user_prompt, question):
    final_prompt = user_prompt + "Question:\n" + question
    response = chat_bedrock.invoke(final_prompt)
    return response.content


def generate_response(prompt):
    elasticsearch_results = get_elasticsearch_results(prompt)
    context_prompt = create_llm_prompt(elasticsearch_results)
    llm_completion = generate_llm_completion(context_prompt, prompt)
    return llm_completion


if __name__ == "__main__":
    st.title("Helios Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Generating response..."):
            chat_history = ""

            if "messages" in st.session_state:
                messages = st.session_state.messages
                if len(messages) > 2:
                    chat_history += "Chat History:\n"
                    for i in range(1, len(messages)-1, 2):
                        if i+1 < len(messages):
                            user_msg = messages[i]["content"]
                            assistant_msg = messages[i+1]["content"]
                            chat_history += f"Previous Question: {user_msg}\n"
                            chat_history += f"Respective Answer: {assistant_msg}\n\n"

            prompt += chat_history

            response = generate_response(prompt)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
