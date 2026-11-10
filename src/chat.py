"""CLI de chat: lê perguntas do usuário e responde com base no conteúdo do PDF."""

from search import answer_question

EXIT_COMMANDS = {"sair", "exit", "quit", "q"}


def main() -> None:
    print("=== Chat RAG - perguntas respondidas com base no PDF ===")
    print("Digite sua pergunta (ou 'sair' para encerrar).")
    print()

    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue

        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = answer_question(question)
        except Exception as error:
            print(f"RESPOSTA: [erro] {error}")
            print()
            continue

        print(f"RESPOSTA: {answer}")
        print()

    print("Chat encerrado.")


if __name__ == "__main__":
    main()
