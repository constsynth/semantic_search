import gradio as gr
from searching_engine import Searcher
from summarize import Summarizer
from pypdf import PdfReader


searcher = Searcher(model_name_or_path='intfloat/multilingual-e5-base',
                    number_of_data=5000,
                    create_database=True)

summarizer = Summarizer(model_name_or_path='IlyaGusev/mbart_ru_sum_gazeta')


def clean_database() -> str:
    searcher.documents = []
    searcher.document_embeddings = None
    return 'База данных успешно очищена'


def add_new_file(pdf_file_name) -> str:
    reader = PdfReader(pdf_file_name)
    doc_text = ''
    for page in reader.pages:
        extracted_text = page.extract_text()
        doc_text += extracted_text
    searcher.update_docs(new_doc=doc_text)
    return 'База данных успешно обновлена'


def search(query, first_n_closest,
           use_summary):
    results = searcher.find_closest_results(query=query, first_n_closest=first_n_closest)
    if use_summary:
        results_with_summaries = []
        for result in results:
            summary = summarizer.summarize(text=result,
                                           max_input=512,
                                           max_length=64,
                                           min_length=32,
                                           no_repeat_ngram_size=4,
                                           temperature=0.2)
            results_with_summaries.append(f'{results.index(result)+1}) Краткая информация: {summary}\n\n'
                                          f'Текст: {result}')
        return '\n\n\n\n'.join(results_with_summaries)
    else:
        results = [f'{results.index(result)+1}) {result}' for result in results]
        return '\n\n\n\n'.join(results)


search = gr.Interface(
    fn=search,
    inputs=[gr.Textbox(label='Найти новость по запросу'),
            gr.Slider(label='Количество новостей',
                      value=5,
                      maximum=10,
                      minimum=1,
                      step=1),
            gr.Checkbox(label='Сделать суммаризацию', value=False)],
    outputs=[gr.Text(label=f'Релевантные новости')],
    allow_flagging='never',
    api_name='search'
)

add_to_database = gr.Interface(
    fn=add_new_file,
    inputs=gr.File(label='Загрузите свой документ', file_types=['.pdf']),
    outputs=gr.Textbox(label='Статус внесенных изменений'),
    allow_flagging='never'
)

delete_from_database = gr.Interface(
    fn=clean_database,
    inputs=None,
    submit_btn=gr.Button('Очистить базу данных', variant='primary'),
    allow_flagging='never',
    outputs=gr.Textbox(label='Статус внесенных изменений')
)

app = gr.TabbedInterface([search, add_to_database, delete_from_database],
                         ['Поиск', 'Добавить в базу данных', 'Очистить базу данных'],
                         theme=gr.themes.Soft())


if __name__ == '__main__':
    app.launch(
        share=False,
        server_name='0.0.0.0',
        server_port=8080
    )
