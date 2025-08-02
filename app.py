import os
import uuid
import requests
import time
import json

# --- Importações Essenciais ---
from flask import Flask, request, jsonify
from twilio.rest import Client

# --- Importações da LangChain e Google Generative AI ---
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


app = Flask(__name__)

@app.route('/')
def health_check():
    """Endpoint simples para verificação de saúde da plataforma."""
    return "OK", 200

# --- CONFIGURAÇÕES IMPORTANTES ---
# É altamente recomendável usar variáveis de ambiente em vez de colar as chaves aqui.
os.environ["GOOGLE_API_KEY"] = "AIzaSyAZcLXqpNirS5DhsYFwWkSAMju24eIFJl0"
TWILIO_ACCOUNT_SID = "ACfc3823b8084d0fd4312885a381db8fe8"
TWILIO_AUTH_TOKEN = "0132c8178dd6db012c3d08323c00b378"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"

# Dicionário global para gerenciar as sessões dos usuários (carregado na inicialização)
user_sessions = {}


# --- Funções de Persistência de Sessão ---

def save_sessions():
    """Salva o dicionário de sessões em um arquivo JSON."""
    with open('sessions.json', 'w') as f:
        json.dump(user_sessions, f, indent=4)
    print("Sessões salvas em sessions.json.")


def load_sessions():
    """Carrega o dicionário de sessões de um arquivo JSON."""
    global user_sessions
    sessions_file = 'sessions.json'

    # Garante que as pastas e o arquivo de sessão existam
    if not os.path.exists("faiss_indices"):
        os.makedirs("faiss_indices")
    if not os.path.exists("temp_pdfs"):
        os.makedirs("temp_pdfs")
    if not os.path.exists(sessions_file):
        with open(sessions_file, "w") as f:
            f.write("{}")

    # Verifica se o arquivo não está vazio antes de tentar carregar
    if os.path.getsize(sessions_file) > 0:
        try:
            with open(sessions_file, 'r') as f:
                user_sessions = json.load(f)
            print(f"Sessões carregadas de sessions.json. {len(user_sessions)} usuários ativos.")
        except json.JSONDecodeError:
            print("Erro ao decodificar sessions.json. O arquivo pode estar corrompido. Iniciando com sessões vazias.")
            user_sessions = {}
    else:
        print("Arquivo de sessão vazio. Iniciando com sessões vazias.")


# --- Funções de Processamento, Download e Envio ---

def processar_pdf_e_criar_qa_chain_user(caminho_pdf, user_waid):
    """
    Carrega um PDF, o processa e salva um índice FAISS local para o usuário.
    Retorna uma tupla (sucesso, resultado), onde resultado é o caminho do índice ou uma mensagem de erro.
    """
    print(f"Iniciando processamento do PDF para o usuário {user_waid}...")
    try:
        loader = PyMuPDFLoader(caminho_pdf)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(data)
        print(f"Documento dividido em {len(docs)} pedaços para o usuário {user_waid}.")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.from_documents(docs, embeddings)

        user_faiss_dir = os.path.join("faiss_indices", user_waid)
        if not os.path.exists(user_faiss_dir):
            os.makedirs(user_faiss_dir)

        faiss_index_path = os.path.join(user_faiss_dir, "faiss_index")
        db.save_local(faiss_index_path)

        print(f"Índice FAISS salvo com sucesso em '{faiss_index_path}' para o usuário {user_waid}.")
        return True, faiss_index_path
    except Exception as e:
        error_message = f"Erro crítico ao processar PDF para {user_waid}: {e}"
        print(error_message)
        return False, error_message


def download_file(url, destination_folder="temp_pdfs", filename=None):
    """
    Baixa um arquivo de uma URL, autenticando com as credenciais da Twilio.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if filename is None:
        filename = url.split('/')[-1]

    file_path = os.path.join(destination_folder, filename)
    try:
        response = requests.get(url, stream=True, auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN))
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Arquivo baixado com sucesso: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar arquivo '{url}': {e}")
        return None


def split_message(text, max_length=1550):
    """
    Divide um texto longo em partes menores para o limite do WhatsApp.
    """
    parts = []
    if not text:
        return parts

    while len(text) > max_length:
        split_point = text.rfind('\n', 0, max_length)
        if split_point == -1:
            split_point = text.rfind('. ', 0, max_length)
        if split_point == -1:
            split_point = text.rfind(' ', 0, max_length)
        if split_point == -1:
            split_point = max_length

        parts.append(text[:split_point + 1])
        text = text[split_point + 1:].lstrip()

    parts.append(text)
    return [part for part in parts if part.strip()]


def send_whatsapp_message(to_phone_number, message_text):
    """
    Envia uma mensagem de texto para um número de telefone via Twilio.
    """
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        if not to_phone_number.startswith('whatsapp:'):
            twilio_to_number = f"whatsapp:{to_phone_number}"
        else:
            twilio_to_number = to_phone_number

        print(f"Enviando mensagem para {twilio_to_number}: '{message_text[:80]}...'")
        message = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_text,
            to=twilio_to_number
        )
        print(f"Mensagem enviada com SID: {message.sid}")
    except Exception as e:
        print(f"Erro ao enviar mensagem via Twilio para {to_phone_number}: {e}")


def get_qa_chain_for_user(user_waid):
    """
    Carrega o índice FAISS de um usuário e monta a cadeia de QA sob demanda.
    """
    if user_waid in user_sessions and user_sessions[user_waid].get("faiss_index_path"):
        faiss_index_path = user_sessions[user_waid]["faiss_index_path"]
        if not os.path.exists(faiss_index_path):
            print(f"Erro: Caminho do índice FAISS não encontrado para o usuário {user_waid}: {faiss_index_path}")
            return None
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

            template = """Você é um assistente de IA especialista em análise de documentos.
            Sua tarefa é responder perguntas baseando-se **exclusivamente** no conteúdo do documento fornecido (contexto).
            Se a resposta não estiver no texto, diga "Essa informação não se encontra no documento.".
            Se a pergunta for um resumo, sintetize os pontos principais do texto.
            Não invente informações fora do contexto.

            Contexto do documento:
            {context}

            Pergunta do usuário:
            {question}

            Resposta detalhada:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=db.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
        except Exception as e:
            print(f"Erro ao carregar QA Chain para o usuário {user_waid}: {e}")
            return None
    return None


def get_user_pdf_name(user_waid):
    """
    Retorna o nome do arquivo PDF associado à sessão de um usuário.
    """
    return user_sessions.get(user_waid, {}).get("pdf_name", None)


# --- ENDPOINT PRINCIPAL: WEBHOOK DA TWILIO ---
@app.route('/webhook', methods=['POST'])
def webhook():
    # Carrega as sessões no início de cada requisição
    load_sessions()

    data = request.form
    print(f"Mensagem recebida do WhatsApp (Twilio Payload): {data}")

    sender_phone = data.get('From')
    text_content = data.get('Body', '').strip()
    media_url = data.get('MediaUrl0')
    num_media = int(data.get('NumMedia', 0))

    if not sender_phone:
        return jsonify({"status": "Ignored payload without sender"}), 200

    if sender_phone.startswith('whatsapp:'):
        sender_phone = sender_phone.replace('whatsapp:', '')

    if sender_phone == TWILIO_WHATSAPP_NUMBER.replace('whatsapp:', ''):
        return jsonify({"status": "Ignored message from self"}), 200

    user_waid = data.get('WaId', sender_phone)  # Usa WaId como ID, com fallback para o número

    message_category = 'document' if num_media > 0 and media_url else 'text'
    print(f"Webhook: User WAID: {user_waid}. Categoria da Mensagem: {message_category}.")

    # CASO 1: Usuário envia um documento
    if message_category == 'document':
        send_whatsapp_message(sender_phone, "Recebi seu documento. Processando para análise, por favor aguarde...")

        original_file_name = media_url.split('/')[-1].split('?')[0] if '?' in media_url else media_url.split('/')[-1]
        unique_filename = f"{uuid.uuid4()}_{original_file_name}"

        caminho_pdf_baixado = download_file(media_url, filename=unique_filename)

        if caminho_pdf_baixado:
            sucesso, resultado = processar_pdf_e_criar_qa_chain_user(caminho_pdf_baixado, user_waid)
            if sucesso:
                # Salva o estado da sessão para este usuário e persiste no arquivo
                user_sessions[user_waid] = {"faiss_index_path": resultado, "pdf_name": original_file_name}
                save_sessions()  # Salva as sessões em disco
                resposta_bot = f"Seu documento '{original_file_name}' foi processado com sucesso! ✅\n\nAgora, pode me fazer perguntas sobre o conteúdo dele."
            else:
                resposta_bot = f"Erro ao processar o seu PDF. Por favor, tente enviar outro arquivo. Detalhe: {resultado}"
            os.remove(caminho_pdf_baixado)
        else:
            resposta_bot = "Não consegui baixar seu documento. Por favor, verifique o link ou tente novamente."

        send_whatsapp_message(sender_phone, resposta_bot)

    # CASO 2: Usuário envia uma pergunta em texto
    elif message_category == 'text':
        if user_waid not in user_sessions or not get_qa_chain_for_user(user_waid):
            resposta_bot = "Olá! 👋 Para começar, por favor, me envie o documento PDF sobre o qual você gostaria de conversar."
        else:
            qa_chain = get_qa_chain_for_user(user_waid)
            if qa_chain:
                try:
                    print(f"Webhook: Enviando pergunta para a IA de {user_waid}: '{text_content}'")
                    resposta_completa_ia = qa_chain.invoke({"query": text_content})["result"]
                    print(f"Webhook: Resposta da IA gerada: '{resposta_completa_ia[:100]}...'")

                    partes_resposta = split_message(resposta_completa_ia)
                    total_parts = len(partes_resposta)

                    for i, part in enumerate(partes_resposta):
                        part_message = f"(Parte {i + 1}/{total_parts})\n{part}" if total_parts > 1 else part
                        send_whatsapp_message(sender_phone, part_message)
                        if total_parts > 1:
                            time.sleep(1.2)
                    return jsonify({"status": "ok, multipart message sent"})

                except Exception as e:
                    resposta_bot = f"Desculpe, ocorreu um erro ao processar sua pergunta: {e}"
                    print(f"Webhook: Erro durante a chamada à QA_Chain: {e}")
            else:
                resposta_bot = "Ocorreu um problema ao carregar os dados do seu PDF. Por favor, envie o documento novamente para recriar o índice."

        send_whatsapp_message(sender_phone, resposta_bot)

    else:
        send_whatsapp_message(sender_phone,
                              "Não entendi sua mensagem. Envie um PDF ou faça uma pergunta sobre um documento já enviado.")

    return jsonify({"status": "ok"})
