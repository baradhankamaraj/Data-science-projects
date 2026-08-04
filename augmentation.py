import json
import re
from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
from vectorstore import vectordb
from llm import llm_call



retriever = vectordb.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.5,
        "k": 3
    }
)

def is_scheme_question(question):

    messages = [
        {
            "role": "system",
            "content": """
        You are a classifier.

        Determine whether the user's question is specifically about Indian Government Schemes.

        Return ONLY one word:

        YES
        or
        NO

        Return NO if the question is about:
        - Prime Minister
        - President
        - Ministers
        - Politics
        - Current affairs
        - History
        - Geography
        - General knowledge
        - Anything not directly asking about a government scheme.
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    response = llm_call(
        messages,
        temperature=0,
        max_tokens=5
    )

    return response.strip().upper() == "YES"

def self_rag(question: str):

    # print("=" * 60)
    # print("Question:", question)

    # ======================================================
    # Scope Check
    # ======================================================

    if not is_scheme_question(question):
        return {
            "Question": question,
            "Answer": "I can answer only questions related to the Government Scheme documents."
        }

    # ======================================================
    # 1. Retrieve Documents
    # ======================================================

    # print("\nRetrieving documents...")

    docs = retriever.invoke("query: " + question)

    if not docs:
        return {
    "Question": question,
    "Answer": "I cannot answer this from the provided documents.",
    "Evaluation": {},
    "Decision": "NO_DOCUMENTS"
}
    # print(f"Retrieved {len(docs)} documents.")

    # ======================================================
    # 2. Build Context
    # ======================================================

    context = ""

    for i, doc in enumerate(docs, start=1):

        context += f"""
        ==============================
        Document {i}
        
        Source:
        {doc.metadata.get("source","Unknown")}
        
        Content:
        {doc.page_content}
        
        ==============================
        
        """
    # print("=" * 80)
    # print("CONTEXT SENT TO LLM")
    # print(context)
    # print("=" * 80)

    # ======================================================
    # 3. Answer Generation
    # ======================================================

    messages = [
    {
        "role": "system",
        "content": (
            """
            You are a Government Scheme Assistant.

            Answer the user's question ONLY using the provided context.
            
            Instructions:
            - Answer the specific question asked.
            - Include all information from the context that is directly relevant to the question.
            - If the answer requires multiple facts from different documents, combine them into a single complete answer.
            - Do not omit relevant information.
            - Do not use your own knowledge.
            - Do not guess.
            - Do not infer.
            - Do not include information unrelated to the question.
            - Do not invent, assume, or infer information that is not explicitly stated in the context.
            - If the context does not contain enough information to answer the question, reply exactly:
              I cannot answer this from the provided documents.
            
            """
        )
    },
    {
        "role": "user",
        "content": f"""
        Context:
        {context}
        
        Question:
        {question}
        """
            }
        ]


    # print("Generating answer...")

    try:
        response = llm_call(
        messages,
        temperature=0.1,
        max_tokens=256
    )
        # print("="*50)
        # print("ANSWER SENT TO CRITIQUE")
        # print(response)
        # print("="*50)
        # response = re.split(r"\n\s*(Question:|Explanation:|Note:)",response)[0].strip()
    except Exception as e:
        return {
    "Question": question,
    "Answer": f"Error: {e}",
    "Evaluation": {},
    "Decision": "ERROR"
}
    

    # ======================================================
    # 4. Self Critique
    # ======================================================
    
    critique_messages = [
        {
        "role": "system",
        "content": ("""
        You are a strict RAG evaluator.

        Evaluate ONLY using the provided context.

        Return ONLY valid JSON.

        Do NOT explain.
        Do NOT use markdown.
        Do NOT use code fences.
        Do NOT write any extra text.

        Output exactly in this format:
                
        {
        "Faithfulness": <1-5>,
        "Completeness": <1-5>,
        "Relevance": <1-5>,
        "Conciseness": <1-5>,
        "Safety": <1-5>
    }""")},

        {
            "role": "user",
            "content": f"""
            Evaluate the following RAG answer.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            {response}
            Score the answer from 1 to 5 for:

            - Faithfulness
            - Completeness
            - Relevance
            - Conciseness
            - Safety

            Return ONLY JSON.
            
            
            """}]
 
    
    
    # print("Evaluating answer...")
    
    try:
        critique_response = llm_call(
        critique_messages,
        temperature=0,
        max_tokens=40)
        # Remove Markdown code fences if present
        critique_response = critique_response.replace("```json", "")
        critique_response = critique_response.replace("```", "")
        critique_response = critique_response.strip()


        # print("=" * 50)
        # print("TYPE:", type(critique_response))
        # print("=" * 50)
        # print(repr(critique_response))
        # print("=" * 50)
    except Exception as e:
        print("Critique LLM Error:")
        print(e)
        raise
    
    
    
    
    
    
    # ======================================================
    # Extract JSON safely
    # ======================================================
    
    match = re.search(r"\{.*\}", critique_response, re.DOTALL)
    if not match:
        print("No JSON found!")
        print(critique_response)
        return {
    "Question": question,
    "Answer": response,
    "Evaluation": {},
    "Decision": "CRITIQUE_FAILED"
}
    try:
        scores = json.loads(match.group())
    except json.JSONDecodeError as e:
        print("Invalid JSON")
        print(e)
        print(match.group())
        return

    
    
    # ======================================================
    # Get scores
    # ======================================================
    
    faithfulness = scores.get(
        "Faithfulness",
        0
    )
    
    completeness = scores.get(
        "Completeness",
        0
    )
    
    relevance = scores.get(
        "Relevance",
        0
    )
    
    conciseness = scores.get(
        "Conciseness",
        0
    )
    
    safety = scores.get(
        "Safety",
        0
    )
    
    
    
    # print("Scores:")
    # print(scores)
    
    
    
    # ======================================================
    # Self-RAG Decision
    # ======================================================
    
    if (
        faithfulness >= 4
        and relevance >= 4
        and safety >= 4
    ):
    
        decision = "ACCEPT"
    
    
    elif (
        relevance <= 2
        or faithfulness <= 2
    ):
    
        decision = "RETRIEVE AGAIN"
    
    
    else:
    
        decision = "REVISE"



   

   

    # -----------------------------
    # Action
    # -----------------------------
    if decision == "ACCEPT":

        return {
            "Question": question,
    
            "Answer": response,
    
#             "Source Documents": list(dict.fromkeys(
#     os.path.basename(doc.metadata.get("source", "Unknown"))
#     for doc in docs
# )),
    
            "Evaluation": {
                "Faithfulness": faithfulness,
                "Completeness": completeness,
                "Relevance": relevance,
                "Conciseness": conciseness,
                "Safety": safety
            },
    
            "Decision": decision
        }

    elif decision == "REVISE":

        revise_message = [
                {
                    "role": "system",
                    "content": (
                        """
                    Rewrite the answer.
                    
                    Rules:
                    
                    1. Use ONLY the supplied context.
                    2. Remove unsupported statements.
                    3. Do NOT add new information.
                    4. Mention the source document. """)
                },
                {
                    "role": "user",
                    "content": f"""

        
        Context:
        {context}
        
        Question:
        {question}
        
        Previous Answer:
        {response}
        
        Revised Answer:
        """
                }]

        revised_answer = llm_call(
    revise_message,
    temperature=0.1,
    max_tokens=512
)

        return {

        "Question": question,

        "Answer": revised_answer,

        "Source Documents": [
            doc.metadata.get("source", "Unknown")
            for doc in docs
        ],

        "Evaluation": {
            "Faithfulness": faithfulness,
            "Completeness": completeness,
            "Relevance": relevance,
            "Conciseness": conciseness,
            "Safety": safety
        },

        "Decision": decision
    }

    else:

        better_docs = retriever.invoke(
        question + " provide detailed information")

        return {
            "Question": question,
    
            "Answer": "Retrieved again with improved query",
    
            "Retrieved Documents": [
                {
                    "Source": doc.metadata.get("source", "Unknown"),
                    "Content": doc.page_content[:500]
                }
                for doc in better_docs
            ],
    
            "Decision": decision
        }
