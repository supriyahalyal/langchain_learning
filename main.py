from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()


def main():
    print("Hello from langchain-learning!")

    information = """
    Girish Karnad (19 May 1938 – 10 June 2019)[1] was an Indian playwright, actor, film director, Kannada writer,[2] and a Jnanpith awardee, who predominantly worked in Kannada, Hindi, Tamil, Telugu, Malayalam and Marathi films. His rise as a playwright in the 1960s marked the coming of age of modern Indian playwriting in Kannada, just as Badal Sarkar did in Bengali, Vijay Tendulkar in Marathi, and Mohan Rakesh in Hindi.[3] He was a recipient of the 1998 Jnanpith Award, the highest literary honour conferred in India.[4]

    For four decades Karnad composed plays, often using history and mythology to tackle contemporary issues. He translated his plays into English and received acclaim. His plays have been translated into some Indian languages and directed by directors like Ebrahim Alkazi, B. V. Karanth, Alyque Padamsee, Prasanna, Arvind Gaur, Satyadev Dubey, Vijaya Mehta, Shyamanand Jalan, Amal Allanaa and Zafer Mohiuddin.[5]

    He was active in the world of Indian cinema working as an actor, director and screenwriter, in Hindi and Kannada cinema, and has earned awards.

    He was conferred Padma Shri and Padma Bhushan by the Government of India and won four Filmfare Awards, of which three are Filmfare Award for Best Director – Kannada and the fourth a Filmfare Best Screenplay Award. He was a presenter for a weekly science magazine programme called "Turning Point" that aired on Doordarshan in 1991.
    """

    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    # llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.5-flash")
    # llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-3.1-pro-preview")
    # llm = ChatOllama(temperature=0, model="qwen3.5:0.8b")
    llm = ChatOllama(temperature=0, model="gemma3:270m")

    chain = summary_prompt_template | llm
    resp = chain.invoke(input={"information": information})

    print(resp.content)



if __name__ == "__main__":
    main()
