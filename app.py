# import os

# import six
# # from google.cloud import translate_v2 as translate
# import streamlit as st
# from dotenv import load_dotenv
# from googletrans import Translator

# import cohere

# load_dotenv()

# translator = Translator()
# co = cohere.Client(os.environ['COHERE_API_KEY'])

# langs = {
#     "af": "Afrikaans",
#     "sq": "Albanian",
#     "ar": "Arabic",
#     "hy": "Armenian",
#     "az": "Azerbaijani",
#     "eu": "Basque",
#     "be": "Belarusian",
#     "bn": "Bengali",
#     "bs": "Bosnian",
#     "bg": "Bulgarian",
#     "ca": "Catalan",
#     "ceb": "Cebuano",
#     "ny": "Chichewa",
#     "zh-cn": "Chinese Simplified",
#     "zh-tw": "Chinese Traditional",
#     "co": "Corsican",
#     "hr": "Croatian",
#     "cs": "Czech",
#     "da": "Danish",
#     "nl": "Dutch",
#     "en": "English",
#     "eo": "Esperanto",
#     "et": "Estonian",
#     "tl": "Filipino",
#     "fi": "Finnish",
#     "fr": "French",
#     "fy": "Frisian",
#     "gl": "Galician",
#     "ka": "Georgian",
#     "de": "German",
#     "el": "Greek",
#     "gu": "Gujarati",
#     "ht": "Haitian Creole",
#     "ha": "Hausa",
#     "haw": "Hawaiian",
#     "iw": "Hebrew",
#     "hi": "Hindi",
#     "hmn": "Hmong",
#     "hu": "Hungarian",
#     "is": "Icelandic",
#     "ig": "Igbo",
#     "id": "Indonesian",
#     "ga": "Irish",
#     "it": "Italian",
#     "ja": "Japanese",
#     "jw": "Javanese",
#     "kn": "Kannada",
#     "kk": "Kazakh",
#     "km": "Khmer",
#     "ko": "Korean",
#     "ku": "Kurdish (Kurmanji)",
#     "ky": "Kyrgyz",
#     "lo": "Lao",
#     "la": "Latin",
#     "lv": "Latvian",
#     "lt": "Lithuanian",
#     "lb": "Luxembourgish",
#     "mk": "Macedonian",
#     "mg": "Malagasy",
#     "ms": "Malay",
#     "ml": "Malayalam",
#     "mt": "Maltese",
#     "mi": "Maori",
#     "mr": "Marathi",
#     "mn": "Mongolian",
#     "my": "Myanmar (Burmese)",
#     "ne": "Nepali",
#     "no": "Norwegian",
#     "ps": "Pashto",
#     "fa": "Persian",
#     "pl": "Polish",
#     "pt": "Portuguese",
#     "ma": "Punjabi",
#     "ro": "Romanian",
#     "ru": "Russian",
#     "sm": "Samoan",
#     "gd": "Scots Gaelic",
#     "sr": "Serbian",
#     "st": "Sesotho",
#     "sn": "Shona",
#     "sd": "Sindhi",
#     "si": "Sinhala",
#     "sk": "Slovak",
#     "sl": "Slovenian",
#     "so": "Somali",
#     "es": "Spanish",
#     "su": "Sudanese",
#     "sw": "Swahili",
#     "sv": "Swedish",
#     "tg": "Tajik",
#     "ta": "Tamil",
#     "te": "Telugu",
#     "th": "Thai",
#     "tr": "Turkish",
#     "uk": "Ukrainian",
#     "ur": "Urdu",
#     "uz": "Uzbek",
#     "vi": "Vietnamese",
#     "cy": "Welsh",
#     "xh": "Xhosa",
#     "yi": "Yiddish",
#     "yo": "Yoruba",
#     "zu": "Zulu"
# }

# options = list(langs.values())

# st.title("MultiLingo:  Multilanguage Text Summarization for Everyone")

# uploaded_file = st.file_uploader(
#     "Upload the txt file to summarize", type="txt")

# selectedLanguage = st.multiselect(
#     "Select a language", options, default=None, max_selections=1)


# def translate_text(target, text):

#     # translate_client = translate.Client()

#     if isinstance(text, six.binary_type):
#         text = text.decode("utf-8")
#     result = translator.translate(text, dest=target)
#     st.download_button('Download summarized text', result.text, file_name='summarized.txt', mime='text/plain')


# def summarize():
#     if uploaded_file is not None and selectedLanguage.__len__() > 0:
#         selectedLanguageKey = list(langs.keys())[list(
#             langs.values()).index(selectedLanguage[0])]
#         summarizeText(selectedLanguageKey)


# def summarizeText(selectedLanguageKey):
#     bytes_data = uploaded_file.getvalue()
#     # st.write(bytes_data)
#     converted_data = bytes_data.decode("utf-8")
#     response = co.summarize(
#         text=converted_data,
#         length='long',
#         format='paragraph',
#         model='summarize-xlarge',
#         additional_command='',
#         temperature=0.3,
#     )
#     translate_text(selectedLanguageKey, response.summary)


# submit_btn = st.button("Summarize", on_click=summarize)

############# Version 2 ###################

# import os

# import six
# # from google.cloud import translate_v2 as translate
# import streamlit as st
# from dotenv import load_dotenv
# from googletrans import Translator

# import cohere

# load_dotenv()

# translator = Translator()
# co = cohere.Client(os.environ['COHERE_API_KEY'])

# langs = {
#     "af": "Afrikaans",
#     "sq": "Albanian",
#     "ar": "Arabic",
#     "hy": "Armenian",
#     "az": "Azerbaijani",
#     "eu": "Basque",
#     "be": "Belarusian",
#     "bn": "Bengali",
#     "bs": "Bosnian",
#     "bg": "Bulgarian",
#     "ca": "Catalan",
#     "ceb": "Cebuano",
#     "ny": "Chichewa",
#     "zh-cn": "Chinese Simplified",
#     "zh-tw": "Chinese Traditional",
#     "co": "Corsican",
#     "hr": "Croatian",
#     "cs": "Czech",
#     "da": "Danish",
#     "nl": "Dutch",
#     "en": "English",
#     "eo": "Esperanto",
#     "et": "Estonian",
#     "tl": "Filipino",
#     "fi": "Finnish",
#     "fr": "French",
#     "fy": "Frisian",
#     "gl": "Galician",
#     "ka": "Georgian",
#     "de": "German",
#     "el": "Greek",
#     "gu": "Gujarati",
#     "ht": "Haitian Creole",
#     "ha": "Hausa",
#     "haw": "Hawaiian",
#     "iw": "Hebrew",
#     "hi": "Hindi",
#     "hmn": "Hmong",
#     "hu": "Hungarian",
#     "is": "Icelandic",
#     "ig": "Igbo",
#     "id": "Indonesian",
#     "ga": "Irish",
#     "it": "Italian",
#     "ja": "Japanese",
#     "jw": "Javanese",
#     "kn": "Kannada",
#     "kk": "Kazakh",
#     "km": "Khmer",
#     "ko": "Korean",
#     "ku": "Kurdish (Kurmanji)",
#     "ky": "Kyrgyz",
#     "lo": "Lao",
#     "la": "Latin",
#     "lv": "Latvian",
#     "lt": "Lithuanian",
#     "lb": "Luxembourgish",
#     "mk": "Macedonian",
#     "mg": "Malagasy",
#     "ms": "Malay",
#     "ml": "Malayalam",
#     "mt": "Maltese",
#     "mi": "Maori",
#     "mr": "Marathi",
#     "mn": "Mongolian",
#     "my": "Myanmar (Burmese)",
#     "ne": "Nepali",
#     "no": "Norwegian",
#     "ps": "Pashto",
#     "fa": "Persian",
#     "pl": "Polish",
#     "pt": "Portuguese",
#     "ma": "Punjabi",
#     "ro": "Romanian",
#     "ru": "Russian",
#     "sm": "Samoan",
#     "gd": "Scots Gaelic",
#     "sr": "Serbian",
#     "st": "Sesotho",
#     "sn": "Shona",
#     "sd": "Sindhi",
#     "si": "Sinhala",
#     "sk": "Slovak",
#     "sl": "Slovenian",
#     "so": "Somali",
#     "es": "Spanish",
#     "su": "Sudanese",
#     "sw": "Swahili",
#     "sv": "Swedish",
#     "tg": "Tajik",
#     "ta": "Tamil",
#     "te": "Telugu",
#     "th": "Thai",
#     "tr": "Turkish",
#     "uk": "Ukrainian",
#     "ur": "Urdu",
#     "uz": "Uzbek",
#     "vi": "Vietnamese",
#     "cy": "Welsh",
#     "xh": "Xhosa",
#     "yi": "Yiddish",
#     "yo": "Yoruba",
#     "zu": "Zulu"
# }

# options = list(langs.values())

# st.title("MultiLingo:  Multilanguage Text Summarization for Everyone")

# uploaded_file = st.file_uploader(
#     "Upload the txt file to summarize", type="txt")

# selectedLanguage = st.multiselect(
#     "Select a language", options, default=None, max_selections=1)

# translated_response = st.empty()  # Create an empty component to display the translated response

# def translate_text(target, text):
#     if isinstance(text, six.binary_type):
#         text = text.decode("utf-8")
#     result = translator.translate(text, dest=target)
#     st.download_button('Download summarized text', result.text, file_name='summarized.txt', mime='text/plain')
#     translated_response.markdown(f"**Translated Response:** {result.text}")  # Display the translated response

# def summarize():
#     if uploaded_file is not None and selectedLanguage.__len__() > 0:
#         selectedLanguageKey = list(langs.keys())[list(
#             langs.values()).index(selectedLanguage[0])]
#         summarizeText(selectedLanguageKey)

# def summarizeText(selectedLanguageKey):
#     bytes_data = uploaded_file.getvalue()
#     converted_data = bytes_data.decode("utf-8")
#     response = co.summarize(
#         text=converted_data,
#         length='long',
#         format='paragraph',
#         model='summarize-xlarge',
#         additional_command='',
#         temperature=0.3,
#     )
#     translate_text(selectedLanguageKey, response.summary)


# submit_btn = st.button("Summarize", on_click=summarize)

############## Version 3 ################

import databutton as db
import streamlit as st
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
import os
import random
import textwrap as tr

# Helper functions (You can access them via View Code page of the app)
from text_load_utils import parse_txt, text_to_docs, parse_pdf
from df_chat import user_message, bot_message

# Initialize the Cohere API
cohere_api_key = db.secrets.get(name="COHERE_API_KEY")

# Display the title and information
st.title("Multilingual Chat Bot 🤖")
st.info(
    "For your personal data! Powered by [cohere](https://cohere.com) + [LangChain](https://python.langchain.com/en/latest/index.html) + [Databutton](https://www.databutton.io) "
)

# Create separate session states for document translation and chat
document_translation_prompt = st.session_state.get("document_translation_prompt", None)
direct_chat_prompt = st.session_state.get("direct_chat_prompt", None)

# Upload a file for document translation
uploaded_file = st.file_uploader(
    "**Upload a pdf or txt file:**",
    type=["pdf", "txt"],
)
page_holder = st.empty()

# Initialize the prompt template for document translation
document_translation_prompt_template = """Text: {context}

Question: {question}

Answer the question based on the text provided. If the text doesn't contain the answer, reply that the answer is not available."""

document_translation_PROMPT = PromptTemplate(
    template=document_translation_prompt_template, input_variables=["context", "question"]
)

# Bot UI dump for document translation
# Display message history for document translation
if document_translation_prompt is None:
    document_translation_prompt = [{"role": "system", "content": document_translation_prompt_template}]

for message in document_translation_prompt:
    if message["role"] == "user":
        user_message(message["content"])
    elif message["role"] == "assistant":
        bot_message(message["content"], bot_name="Document Translation Bot")

# Process uploaded file for document translation
if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        doc = parse_txt(uploaded_file)
    else:
        doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)

# Display the file content for document translation
with page_holder.expander("File Content (Document Translation)", expanded=False):
    pages

# Initialize Cohere embeddings and create a vector store for document translation
embeddings = CohereEmbeddings(
    model="multilingual-22-12", cohere_api_key=cohere_api_key
)
document_translation_store = Qdrant.from_documents(
    pages,
    embeddings,
    location=":memory:",
    collection_name="document_translation",
    distance_func="Dot",
)

# Create a container for document translation messages
document_translation_messages_container = st.container()
document_translation_question = st.text_input(
    "", placeholder="Type your message here (Document Translation)", label_visibility="collapsed"
)

if st.button("Run (Document Translation)", type="secondary"):
    document_translation_prompt.append({"role": "user", "content": document_translation_question})
    chain_type_kwargs = {"prompt": document_translation_PROMPT}
    with document_translation_messages_container:
        user_message(document_translation_question)
        botmsg = bot_message("...", bot_name="Document Translation Bot")

    # Perform question answering using RetrievalQA for document translation
    document_translation_qa = RetrievalQA.from_chain_type(
        llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
        chain_type="stuff",
        retriever=document_translation_store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )

    answer = document_translation_qa({"query": document_translation_question})
    result = answer["result"].replace("\n", "").replace("Answer:", "")

    with st.spinner("Loading response .."):
        botmsg.update(result)

    document_translation_prompt.append({"role": "assistant", "content": result})

# Create a new chat section for direct conversation with Cohere
st.title("Direct Chat 📩")

# Initialize the prompt template for direct chat
direct_chat_prompt_template = """Input: {chat_input}

Output: {chat_output}"""

direct_chat_PROMPT = PromptTemplate(
    template=direct_chat_prompt_template, input_variables=["chat_input", "chat_output"]
)

direct_chat_messages_container = st.container()
direct_chat_input = st.text_input(
    "", placeholder="Type your message here (Direct Chat)", label_visibility="collapsed"
)

if st.button("Send (Direct Chat)", type="secondary"):
    direct_chat_prompt.append({"role": "user", "content": direct_chat_input})
    chain_type_kwargs = {"prompt": direct_chat_PROMPT}
    with direct_chat_messages_container:
        user_message(direct_chat_input)
        botmsg = bot_message("...", bot_name="Direct Chat Bot")

    # Perform question answering using RetrievalQA for direct chat
    direct_chat_qa = RetrievalQA.from_chain_type(
        llm=Cohere(model="command", temperature=0, cohere_api_key=cohere_api_key),
        chain_type="stuff",
        retriever=store.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
    )

    chat_output = direct_chat_qa({"chat_input": direct_chat_input})
    result = chat_output["chat_output"]

    with st.spinner("Loading response .."):
        botmsg.update(result)

    direct_chat_prompt.append({"role": "assistant", "content": result})

st.session_state["document_translation_prompt"] = document_translation_prompt
st.session_state["direct_chat_prompt"] = direct_chat_prompt
