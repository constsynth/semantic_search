import gradio as gr
from searching_engine import Searcher
from summarize import Summarizer

searcher = Searcher(model_name_or_path='DeepPavlov/rubert-base-cased', number_of_data=5000)
summarizer = Summarizer(model_name_or_path='IlyaGusev/mbart_ru_sum_gazeta')


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
            results_with_summaries.append(f'{results.index(result)+1}) Краткая информация по новости: {summary}\n\n'
                                          f'Новость: {result}')
        return '\n\n\n\n'.join(results_with_summaries)
    else:
        results = [f'{results.index(result)+1}) {result}' for result in results]
        return '\n\n\n\n'.join(results)


search = gr.Interface(
    theme=gr.themes.Soft(),
    fn=search,
    inputs=[gr.Textbox(label='Найти новость по запросу'),
            gr.Slider(label='Количество новостей',
                      value=5,
                      maximum=10,
                      minimum=1),
            gr.Checkbox(label='Сделать суммаризацию', value=False)],
    outputs=[gr.Text(label=f'Релевантные новости')],
    api_name='search'
)

app = search


if __name__ == '__main__':
    app.launch(
        share=False,
        server_name='0.0.0.0',
        server_port=8080
    )
