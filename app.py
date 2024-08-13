#bibliotecas
import streamlit as st
import pandas as pd
import math
import numpy as nppip
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#para usar o prophet
import logging
from prophet import Prophet


#import joblib 
#import pickle 
import time

# Configurar o layout da página

st.set_page_config(layout="wide")         
                 
col1, col2, col3 = st.columns([1,8,1])
col4,col5, col6 = st.columns([0.5,9,0.5])
col7, col8 = st.columns([7,3]) 
col9,col10 = st.columns([7,3]) 
col11, col12 = st.columns([5,5]) 
col13,col14 = st.columns(2) 
col15,col16 = st.columns(2)
col17, col18 = st.columns(2)   
col19, col20 = st.columns(2)

st.markdown("""
<style>
.big-font {
    font-size:20px !important;            
}
.media-font {
    font-size:20px !important;            
}         
</style>
""", unsafe_allow_html=True)

#título da página

with col1:
    st.write("")
with col2:
    st.title(":orange[ Passos Mágicos - Análise de Dados/PEDE]")  
with col3:
    st.write("")


#imagem inicial


with col4:
    st.write("")
with col5:
    st.image('IMG_passos_magicos.png', width=1000,use_column_width=1, caption='Fonte: https://passosmagicos.org.br/')
with col6:
    st.write("")


#Hisória


st.subheader("", divider="orange")
st.subheader("História da Organização Não Governamental Passos Mágicos", divider="orange") 
with st.container():
    col7, col8 = st.columns([6,4])
    with col7:

        #informações iniciais
        paragrafo1 =""" "A Associação Passos Mágicos tem uma trajetória de 31 anos de atuação, trabalhando na transformação da vida de crianças e jovens baixa renda os levando a melhores oportunidades de vida.

    A transformação, idealizada por Michelle Flues e Dimetri Ivanoff, começou em 1992, atuando dentro de orfanatos, no município de Embu-Guaçu.

    Em 2016, depois de anos de atuação, decidem ampliar o programa para que mais jovens tivessem acesso a essa fórmula mágica para transformação que inclui: educação de qualidade, auxílio psicológico/psicopedagógico, ampliação de sua visão de mundo e protagonismo. Passaram então a atuar como um projeto social e educacional, criando assim a Associação Passos Mágicos." """
        
        st.write(paragrafo1)

        paragrafo2="""

        A Associação realizou a Pesquisa Extensiva do Desenvolvimento Educacional-PEDE durante os anos de 2020 a 2022, a fim de de sistematizar as suas ações sociais e registrar de forma rigorosa seus processos. O PEDE aglutina indicadores e informações que fornecem subsídios para análise de dados e proposição de estratégias de melhoria da promoção educacional dos alunos. 

        Com a presente análise de dados, propormos um olhar diferenciado sobre os dados obtidos, além de trazer proposições que visem aprimorar o grandioso trabalho da ONG a fim de transformar cada vez mais vidas com o seu projeto inovador.

        """
        st.write(paragrafo2)
        st.write(" ")
        st.write(" ")


    with col8:
        video_file = open("video.mp4", "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
        st.write("Fonte: https://passosmagicos.org.br/")

dados = pd.read_csv('df_final_passos.csv', sep=',')

st.write("")
st.subheader("", divider="orange")
st.subheader("Dados da Pesquisa Extensiva do Desenvolvimento Educacional-PEDE dos anos 2020 a 2022 ", divider="orange") 

#Análise de INDE

st.subheader('Análise do INDE')
with st.container():
    col9, col10 = st.columns(2) 
    st.write(" ")  
    with col9:    
        #gráfico1
        fig = plt.figure()
        inde_medio = dados[['ano', 'inde']].groupby('ano')['inde'].mean().round()
       
        sns.lineplot(data=dados, x='nome', y='inde', color='#ADD8E6', weights=1)
        plt.title('Histórico de Registros do Indicador de Desenvolvimento Educacional-INDE')
        plt.axhline(y =dados.inde.mean().round(2),linewidth=1, color='r', label='Média Geral = {:.2f}'.format(dados.inde.mean().round(2)),alpha=0.7 ,linestyle='dashed')
        plt.legend()
        plt.ylabel('valores')
        plt.xlabel('registros')
        plt.xlim(0,1350)
        plt.grid(color='lightgrey', alpha=0.2)
        st.pyplot(fig)
        col1, col2, col3 = st.columns(3)
        col1.metric(label='INDE mínimo', value=dados['inde'].min())  
        col2.metric(label='INDE médio', value=dados['inde'].mean().round(2)) 
        col3.metric(label='INDE máximo', value=dados['inde'].max()) 
        
        
        

    with col10:
        st.image('IMG_inde.png', width=550,use_column_width=0.9, caption='Fonte: https://passosmagicos.org.br/')
        st.write(" ")
        st.image('IMG_calculo_inde.png', width=550,use_column_width=0.9, caption='Fonte: https://passosmagicos.org.br/')
        st.write(" ")

st.write(" ☞ O INDE é formado por 7 indicadores distribuídos nas dimensões acadêmicas, psicossocial e psicopedagógica, são eles: IAN, IDA, IEG, IAA, IPS, IPP e IPV. Observamos no gráfico acima a disposição dos valores de INDE obtidos pelos alunos nos anos de 2020 a 2022 e podemos perceber que houve uma quantidade relevante de notas abaixo da média geral, o que é um idicativo da necessidade de empreender esforços para a melhoria do indicador nos anos subsequentes.")
st.write(" ")
st.write(" ")


#Análide do POnto_Virada


st.subheader('Análise do Ponto de Virada')
with st.container():
    col11, col12 = st.columns(2) 
    with col11:
        #gráfico de viradas
        df_virada_por_ano = dados[["nome","ano","ponto_virada","fase"]].copy()
        df_virada_por_ano  = pd.DataFrame(df_virada_por_ano .query('ponto_virada != "0"'))
        df_virada_por_ano  = pd.DataFrame(df_virada_por_ano .groupby(['ano','fase'])['nome'].count())
        df_virada_por_ano.columns = ['qtd_viradas']
        fig = plt.figure(figsize=(9,6))
        sns.barplot(data =df_virada_por_ano , x='fase', y='qtd_viradas',hue='ano', palette=['lightblue', 'gray', '#005A92'])
        plt.title('Análise da Quantidade de Ponto de Viradas-IPV por Fase', fontsize=14)
        plt.grid(color='lightblue', alpha=0.2)
        plt.legend(title= "PEDE- Ano")
        st.pyplot(fig)

        st.write(" ☞ O Ponto de Virada-IPV é um  indicador muito importante na organização Passos Mágicos da dimensão psicopedagógica. Ele faz referência  a um estágio do desenvolvimento  em que o estudante demonstra de forma ativa, estar consciente da importância da educação, do valor do saber e da importância de aprender. Analisando a quantidade de alunos que obtiveram Ponto de Virada distribuídos pelas fases, observamos uma maior frequência de IPV no ano de 2022 na maioria das fases. É importante salientar que não obtivemos Ponto de Virada nas fase 8 para os anos de 2021 e 2022.")

        st.write("")

    with col12:
        df_virada_por_aluno = dados[["nome","ponto_virada"]]. groupby('nome')['ponto_virada'].sum().value_counts()
        col1, col2, col3= st.columns([0.1,9.8,0.1])
        
        with  col1:
          st.write(" ")  
       
        with col2: 
            fig2 = plt.figure(figsize=(9,6))
            ax =sns.barplot(df_virada_por_aluno, palette='Blues')
            plt.title('Análise da Quantidade de Viradas por Aluno', fontsize=16)
            ax.bar_label(ax.containers[0], fontsize=10)
            ax.bar_label(ax.containers[1], fontsize=10)
            ax.bar_label(ax.containers[2], fontsize=10)
            ax.bar_label(ax.containers[3], fontsize=10)
            plt.grid(alpha=0.2, color='lightblue')
            st.pyplot(fig2)
        with  col3:
          st.write(" ") 
        col4, col5 = st.columns([3,7])
        col4.dataframe(df_virada_por_aluno)
        col5.write('  ☞ Observamos que a maioria dos alunos não obtiveram o indicador Ponto de Virada durante os anos avaliados. Fato esse que requer atenção da organização no intuito de promover ações que visem estimular os alunos para obtenção do índice.')   


#Análise Bolsista


st.subheader('Análise dos Bolsistas')
with st.container():
    df_bolsista_por_fase = dados[['nome','ano','fase', 'bolsista']]
    df_bolsista_por_fase  = pd.DataFrame(df_bolsista_por_fase .query('bolsista != "0"'))
    df_bolsista_por_fase  = pd.DataFrame(df_bolsista_por_fase .groupby(by=['fase','ano'])['nome'].count())
    df_bolsista_por_fase.columns= ['qtd_alunos']

    col13,col14 = st.columns(2) 
     
    with col13:
        fig3 = plt.figure(figsize=(8,6))
        sns.barplot(df_bolsista_por_fase, x='fase', y='qtd_alunos', ci=None, palette='Blues')
        plt.title('Quantidade de Bolsistas por Fase', fontsize=14)
        plt.grid(color='lightblue', alpha=0.3)
        st.pyplot(fig3)

    with col14:

        fig4 = plt.figure(figsize=(8,6))
        sns.barplot(df_bolsista_por_fase, x='fase', y='qtd_alunos', ci=None, palette=['lightblue', 'gray', '#005A92'], hue='ano')
        plt.title('Quantidade de Bolsistas por Fase & Ano', fontsize=14)
        plt.grid(color='lightblue', alpha=0.3)
        st.pyplot(fig4)

st.write(" ☞ A bolsa de estudo é um benefício concedido com base em critérios pré-estabelecidos, tais como, notas, comportamento do aluno, entrega de tarefas, demonstração de esforço pessoal e dedicação, participação nas atividades da ONG, avalição do corpo de profissionais da psicologia, participação em voluntariado etc. Podemos observar que a maior quantidade de bolsas concedidas está distribuída nas fases iniciais, especialmente nas fases 0 a 3.")      

df_medias_bolsista_por_ano = dados[['ano', 'ian', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv', 'inde', 'bolsista']].groupby(['ano','bolsista']).mean(numeric_only=True).round(2)
#st.dataframe(df_medias_bolsista_por_ano)
df = df_medias_bolsista_por_ano.copy()
df.index = pd.MultiIndex.from_tuples([('2020', 'Não Bolsista'),
            ('2020', 'Bolsista'),
            ('2021', 'Não Bolsista'),
            ('2021', 'Bolsista'),
            ('2022', 'Não Bolsista'),
            ('2022', 'Bolsista')], names=['ano', 'bolsista'])

with st.container():
    col15, col16= st.columns([7,3])
    with col15:
        st.write("")

        fig5 = plt.figure(figsize=(15,9))
        ax = sns.barplot(data=df, x='ano',y='inde', hue='bolsista', palette=['lightblue', '#005A92'])
        plt.title("Análise Comparativa das Médias do INDE dos Alunos", fontsize=20)
        ax.bar_label(ax.containers[0], fontsize=12)
        ax.bar_label(ax.containers[1], fontsize=12)
        media = dados['inde'].mean()
        plt.axhline(dados['inde'].mean(), color='red', linestyle='--', alpha=0.5, label='Média Geral= {:.2f}'.format(media))
        plt.legend( bbox_to_anchor=(1.22,1))
        plt.grid(alpha=0.2)
        plt.ylim(0,10)
        st.pyplot(fig5)
    
    with col16:
        st.write("    ")           
        st.write("    ")    
        st.write("    ")    
        st.write("    ")    
        st.write("    ")    
        st.write("    ")           
        st.write("    ")    
        st.write("    ")    
                       
        st.write(" ☞ Observamos que as melhores médias de INDE são dos alunos bolsistas em todos os anos estudados. A médias de INDE de alunos não bolsistas no ano de 2020 ultrapassou a média geral dos alunos, contudo o mesmo não ocorreu em 2021 e 2022,  podendo ser um indicativo da necessidade de intensificar os esforços a fim de melhorar o índice nos anos subsequentes, bem como há um despertar para a necesidade do aumento de alunos que posssam adquirir o benefício da bolsa de estudo.")


#Análise dos indicares PEDE por ano


st.subheader('Análise dos Indicadores do PEDE por Ano')
fig6,axes = plt.subplots(nrows=4, ncols=2, figsize=(15,9), layout="constrained")
ax1 = sns.barplot(x='ano', y='ian', data=df, ax=axes[0][0], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax1.set_ylim(0,10)
ax1.bar_label(ax1.containers[0], fontsize=10)
ax1.bar_label(ax1.containers[1], fontsize=10)

ax2 = sns.barplot(x='ano', y='ida', data=df, ax=axes[0][1], hue='bolsista', palette=['lightblue', 'gray'])
ax2.set_ylim(0,10)
ax2.legend(bbox_to_anchor=(1.25,1), loc='upper right')
ax2.bar_label(ax2.containers[0], fontsize=10)
ax2.bar_label(ax2.containers[1], fontsize=10)

ax3 = sns.barplot(x='ano', y='ieg', data=df, ax=axes[1][0], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax3.set_ylim(0,10)
ax3.bar_label(ax3.containers[0], fontsize=10)
ax3.bar_label(ax3.containers[1], fontsize=10)

ax4 = sns.barplot(x='ano', y='iaa', data=df, ax=axes[1][1], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax4.set_ylim(0,10)
ax4.bar_label(ax4.containers[0], fontsize=10)
ax4.bar_label(ax4.containers[1], fontsize=10)

ax5 = sns.barplot(x='ano', y='ips', data=df, ax=axes[2][0], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax5.set_ylim(0,10)
ax5.bar_label(ax5.containers[0], fontsize=10)
ax5.bar_label(ax5.containers[1], fontsize=10)

ax6 = sns.barplot(x='ano', y='ipp', data=df, ax=axes[2][1], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax6.set_ylim(0,10)
ax6.bar_label(ax6.containers[0], fontsize=10)
ax6.bar_label(ax6.containers[1], fontsize=10)

ax7 = sns.barplot(x='ano', y='ipv', data=df, ax=axes[3][0], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax7.set_ylim(0,10)
ax7.bar_label(ax7.containers[0], fontsize=10)
ax7.bar_label(ax7.containers[1], fontsize=10)

ax8= sns.barplot(x='ano', y='inde', data=df, ax=axes[3][1], hue='bolsista',legend=False, palette=['lightblue', 'gray'])
ax8.set_ylim(0,10)
ax8.bar_label(ax8.containers[0], fontsize=10)
ax8.bar_label(ax8.containers[1], fontsize=10)

plt.tight_layout()
st.pyplot(fig6)

st.write(" ☞ O gráfico acima expõe os índices médios por ano, dispostos em grupos de bolsistas e não bolsistas. Os bolsistas apresentaram melhores pontuações no IAN, IDA, IEG e INDE, em todos os anos avaliados. Os bolsistas, em 2022, tiveram menor  IAA do que os não bolsistas. Os valores médios de IPS  dos não bolsistas foram maiores, em todo o intervalo analisado. Os índices IPP  e IPV registraram, em 2020, uma média menor para os bolsistas, contudo o mesmo não ocorreu nos anos seguintes.")


#Previsão 
st.write("")
st.subheader("", divider="orange")
st.subheader(' Previsões ', divider='orange')

#1-prevendo o número futuro de ingresso de novos alunos com Prophet
st.markdown('▪ Prevendo a quantidade de novos alunos ingressos para os anos subsequentes')
df_aluno = dados.groupby('ano_ingresso')['nome'].count()
df_aluno = df_aluno.reset_index()
df_aluno['ano_ingresso'] =  ['2016-01-01', '2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01', '2022-01-01']
df_aluno['ano_ingresso'] = pd.to_datetime(df_aluno['ano_ingresso'])
df_aluno.columns = ['ds', 'y']
with st.container():
    col17, col18 = st.columns(2)
    with col17:
        model = Prophet(yearly_seasonality=True)
        model.fit(df_aluno) 
        future = model.make_future_dataframe(periods=7, freq='YS')
        # fazendo o predict
        col1, col2, col3 = st.columns([1,7,1])
        with col1:
            st.write("")
        with col2:
            forecast = model.predict(future)
            df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
            df_forecast['ds'] =  df_forecast['ds'].dt.year
            st.table( df_forecast.set_index('ds').tail(6) )  
        with col3:
            st.write("")
    with col18:
        # plotando as previsões
        st.pyplot(model.plot(forecast))
    st.write(" ☞ Podemos visualizar uma previsão de aumento do número de alunos para os próximos 5 anos, o que pode ser um indicativo para a instituição realizar planejamento de expansão de suas atividades institucionais. ")    

     
#2-prevendo os valores de inde com Linear Regression

st.markdown('▪ Prevendo o coeficiente de incremento do INDE')
with st.container():
    col1, col2 = st.columns([6,4])
    df_inde = dados[['ano', 'inde']]
    X = df_inde[['ano']].values.reshape(-1,1)
    y = df_inde.inde.values.reshape(-1,1)
    # instanciando do modelo
    model = LinearRegression()
    # treinando o modelo
    model.fit(X,y)
    # O coeficiente de 'ANO' em nosso modelo nos dirá a taxa de incremento por ano
    coef_increment_per_year = (model.coef_[0][0]).round(3)
    st.write(f"☞ A regressão linear realizada entre o INDE e o ano indica um coeficiente de aproximadamente {coef_increment_per_year}. Conforme o modelo, isso significa que  existe uma previsão de diminuição do INDE médio por ano, contrariando o incremento observado entre 2021 e 2022.")
   



#Análise individual do aluno


st.write("")
st.subheader("", divider="orange")
st.subheader('Análise dos Indicadores por Aluno ', divider='orange')
with st.container():
    col19, col20= st. columns([4,6])
    with col19:
        st.write(" ☞ A análise individualizada do aluno pode fornecer subsídios aos profissionais da ONG para prover meios de modificar a situação atual do aluno, estimulando os índices deficientes, caso o mesmo apresente dificuldades, ou mesmo para beneficiá-lo com a bolsa de estudo.")  
    with col20:
        #escolhendo o aluno
        st.subheader('Escolha o aluno para avaliar: ')
        max_alunos = len(dados['nome'].unique())
        num = st.slider("Aluno nº ", 1, max_alunos,1)
        lista_alunos = dados['nome'].unique().tolist()
        if num not in lista_alunos:
            st.write('Não temos dados para o aluno.')     
 #barra de progresso
mensagem = "Operação em progresso. Por favor, aguarde."
barra = st.progress(0, text=mensagem)
for i in range(100):
    time.sleep(0.01)
    barra.progress(i+1, text=mensagem)
time.sleep(1)
barra.empty()
#função para plot
def analise_aluno(num):
    #criando o dataframe por aluno 
    df =  pd.DataFrame(dados[['nome','ano_ingresso','bolsista','ano','iaa','ieg','ips','ida','ipp','ipv','ian','inde']])
    df.columns = ['nome','ano_ingresso','bolsista','ano','IAA','IEG','IPS','IDA','IPP','IPV','IAN','INDE']
    df = df.loc[df['nome'] == num]
    df_media_por_aluno = pd.DataFrame(dados[['nome', 'iaa','ieg','ips','ida','ipp','ipv','ian','inde']].groupby(['nome']).mean().round(2))
    df_media_por_aluno = df_media_por_aluno.loc[df_media_por_aluno.index == num]
    df['bolsista'] = df['bolsista'].replace('0','Não Bolsista')
    df['bolsista'] = df['bolsista'].replace('1','Bolsista')
    bolsista = df['bolsista'].max()
    if bolsista == 1:
        bolsista = 'Bolsista'
    else:
        bolsista = 'Não Bolsista'
    ingresso = df['ano_ingresso'].min()
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<p class="media-font">▪ Notas anuais do Aluno ID {num} :</p>', unsafe_allow_html=True)
            st.dataframe(df.set_index('nome')[['ano','IAA','IEG','IPS','IDA','IPP','IPV','IAN','INDE']])      
            st.markdown(f':blue-background[ ☞ O aluno ID {num} é \'{bolsista}\' e ingressou na Passos Mágicos no ano \'{ingresso}]\'')
            st.write("")
            st.write("")
            df_media_por_aluno = dados[['nome', 'iaa','ieg','ips','ida','ipp','ipv','ian','inde']].groupby(['nome']).mean().round(2)
            df_media_por_aluno = pd.DataFrame(df_media_por_aluno.loc[df_media_por_aluno.index == num])
            df_media_por_aluno.columns = ['IAA','IEG','IPS','IDA','IPP','IPV','IAN','INDE']
            lista_col = df_media_por_aluno.columns.tolist()
            st.markdown(f'<p class="media-font">▪ Médias Gerais do Aluno ID {num}:</p>', unsafe_allow_html=True)
            st.dataframe(df_media_por_aluno )
            for i in range(0, 8):
                lista_col = df_media_por_aluno.columns.tolist()
                x_min = df_media_por_aluno.T.iloc[i].values[0]
                if i == 0:
                    df_min = x_min
                    ind_min = lista_col[i] 
                else:
                    if  x_min <= df_min:
                        df_min = x_min
                        ind_min = lista_col[i]        
            st.markdown(f':blue-background[ ☞ O indicador que o aluno obteve menor nota média foi o \'{ind_min}\'com a nota = {df_min}]')
            media = dados['inde'].mean() 
            media_aluno = df_media_por_aluno['INDE'].mean() 
            if media_aluno > media:
                st.success('#### 👏 Excelente! O  INDE MÉDIO do aluno foi maior que a média geral.')
                st.balloons()
            else:
                st.error('#### 👀 O INDE MÉDIO do aluno ficou abaixo da média geral. ')
        with col2:    
            fig_ = plt.figure(figsize= (6,4))
            sns.barplot(df_media_por_aluno, ci=None, palette='Blues')
            plt.title(f'Análise da Média Geral dos Indicadores do Aluno ID {num}')
            plt.xticks(rotation=360)
            plt.xlabel('Indicadores')
            plt.ylim(0,10)
            plt.ylabel('valores')
            plt.grid(color='lightgray', alpha=0.2)
            st.pyplot(fig_)
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    with st.container():
        col1, col2, col3 = st.columns([6,2,2])
        with col1:
            st.markdown(f'<p class="media-font">▪ Análise Comparativa dos Indicadores do Aluno ID {num}</p>', unsafe_allow_html=True)
            st.write("")
        with col2:
            st.write("")
            #st.markdown(f'<p class="big-font">Análise Comparativa dos Indicadores do Aluno ID {num}</p>', unsafe_allow_html=True)
        with col3:
            st.write("")
            
    fig7,axes = plt.subplots(nrows=4, ncols=2, figsize=(15,9), layout="constrained")

    ax1 = sns.barplot(x='ano', y='IAN', data = df, ax=axes[0][0], legend=False, palette=['lightblue','gray'])
    ax1.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax1.bar_label(ax1.containers[i], fontsize=10)

    ax2 = sns.barplot(x='ano',y='IDA', data=df, ax=axes[0][1],  palette=['lightblue', 'gray'])
    ax2.set_ylim(0,10)
    ax2.legend(bbox_to_anchor=(1.25,1), loc='upper right')
    for i in range(0,len(df.index.tolist())):
        ax2.bar_label(ax2.containers[i], fontsize=10)


    ax3 = sns.barplot(x='ano', y='IEG', data=df, ax=axes[1][0], legend=False, palette=['lightblue', 'gray'])
    ax3.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax3.bar_label(ax3.containers[i], fontsize=10)


    ax4 = sns.barplot( x='ano', y='IAA', data=df, ax=axes[1][1], legend=False, palette=['lightblue', 'gray'])
    ax4.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax4.bar_label(ax4.containers[i], fontsize=10)


    ax5 = sns.barplot(x='ano', y='IPS', data=df, ax=axes[2][0], legend=False, palette=['lightblue', 'gray'])
    ax5.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax5.bar_label(ax5.containers[i], fontsize=10)


    ax6 = sns.barplot(x='ano', y='IPP', data=df, ax=axes[2][1], legend=False, palette=['lightblue', 'gray'])
    ax6.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax6.bar_label(ax6.containers[i], fontsize=10)


    ax7 = sns.barplot(x='ano', y='IPV', data=df, ax=axes[3][0], legend=False, palette=['lightblue', 'gray'])
    ax7.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax7.bar_label(ax7.containers[i], fontsize=10)


    ax8= sns.barplot(x='ano',y='INDE', data=df, ax=axes[3][1], legend=False, palette=['lightblue', 'gray'])
    ax8.set_ylim(0,10)
    for i in range(0,len(df.index.tolist())):
        ax8.bar_label(ax8.containers[i], fontsize=10)

    plt.tight_layout()
    st.pyplot(fig7)
    st.write(" ")
    st.write(" ")
    id_aluno = num
    with st.container():
        st.write(" ")
        st.markdown(f'<p class="media-font">▪ Previsões para o Aluno ID {num}:</p>', unsafe_allow_html=True)
        st.write(" ")
        col1, col2 = st.columns([4,6])
        with col1:
            #prevendo o incremento dos indicadores do aluno
            
            fig8 = plt.figure(figsize=(5,4))
            
            df =  pd.DataFrame(dados[['nome','ano','iaa','ieg','ips','ida','ipp','ipv','ian','inde']])
            df = df.set_index('nome')
            df.columns = ['ano','IAA','IEG','IPS','IDA','IPP','IPV','IAN','INDE']
            df_aluno = pd.DataFrame(df.loc[df.index == num])
            #df_aluno = df[df['nome'] == id_aluno][['ano','iaa', 'ieg', 'ips', 'ida', 'ipp','ipv', 'ian', 'inde']]
            df_sem_ano = df_aluno[['IAA','IEG','IPS','IDA','IPP','IPV','IAN','INDE']]
            colunas = df_sem_ano.columns.tolist()

            lista_inc = []

            for col in colunas:
                # Criar um modelo de regressão linear
                model = LinearRegression()

                # Preparar os dados para regressão
                X = df_aluno['ano'].values.reshape(-1, 1)
                y = df_sem_ano[col].values.reshape(-1, 1)

                # Ajuste o modelo
                model.fit(X, y)

                # O coeficiente de 'ANO' em nosso modelo nos dirá a taxa de incremento por ano
                coef_increment_per_year = model.coef_[0][0]
                
                x = coef_increment_per_year.round(2)
                lista_inc.append(x)

            def cor_da_barra(lista_inc: float) -> str:
                cor=[]
                for i in range(len(colunas)):
                    if lista_inc[i] > 0:
                        cor.append('lightblue')
                    else:
                        cor.append("#FA8072")
                return cor
            palette = cor_da_barra(lista_inc)
            sns.barplot(x = colunas, y =lista_inc, palette=palette)
            plt.axhline(y=0, color= 'gray')
            plt.title(f'Previsão de Incremento dos Indicadores do Aluno ID{num}', fontsize=10)
            st.pyplot(fig8)
        with col2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(f"☞  No gráfico ao lado, podemos visualizar a previsão de incremento dos indicadores do PEDE para o aluno ID {id_aluno}. A partir dessa percepção indicativa, poderemos traçar rotas alternativas que viabilizem a melhora individualiza do aluno, bem como evitar que um possível decréscimo  dos índices ocorra. Pelo fato de indicador possuir atributos definidos e determinados para a sua ponderação, isso facilitaria o foco dos profissionais a fim de promover melhora especializada.")
        
analise_aluno(num)

st.markdown('##### Links:')
with st.container():
    col21, col22 = st.columns([3,7])   
    with col21:   
        st.write("☞ ONG Passos Mágicos")
        st.write("")
        #st.write("☞ Streamlit App: ")       
    with col22:
        st.link_button("Site da ONG Passos Mágicos","https://passosmagicos.org.br/",type="secondary")
        #st.link_button("Streamlit App", "/",type="secondary")
