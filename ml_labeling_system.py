"""
Sistema de Rotulação Semi-Supervisionada de Dados Orçamentários

Este sistema implementa:
1. Pré-processamento de texto das descrições orçamentárias
2. Vetorização com TF-IDF (apenas características textuais)
3. Clustering com DBSCAN (vigilância ρ ≥ 0.9) para agrupar itens similares
4. Exportação dos clusters para rotulação manual
5. Aprendizado semi-supervisionado iterativo (após rotulação manual)
6. Visualização e análise de resultados

Fluxo de trabalho:
- Etapa 1: Gerar clusters de alta similaridade
- Etapa 2: Usuário rotula manualmente alguns exemplos de cada cluster
- Etapa 3: Algoritmo propaga os rótulos para dados não rotulados
- Etapa 4: Treinar classificador com base rotulada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Configuração de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SemiSupervisedBudgetLabeler:
    """
    Classe principal para rotulação semi-supervisionada de dados orçamentários
    """
    
    def __init__(self, vigilance=0.9):
        """
        Inicializa o sistema de rotulação
        
        Args:
            vigilance (float): Parâmetro de vigilância (similaridade mínima) para clustering
        """
        self.vigilance = vigilance
        self.df = None
        self.features_matrix = None
        self.labels = None
        self.confidence_scores = None
        self.vectorizer = None
        self.label_mapping = {}
        self.iteration_history = []
        
    def load_data(self, filepath):
        """Carrega os dados do arquivo Excel"""
        print("📊 Carregando dados...")
        self.df = pd.read_excel(filepath)
        self.original_df = self.df.copy()
        print(f"✓ {len(self.df)} registros carregados")
        return self
    
    def preprocess_text(self, text):
        """
        Pré-processa texto para análise
        
        Args:
            text: Texto a ser processado
            
        Returns:
            Texto processado
        """
        if pd.isna(text):
            return ""
        
        text = str(text).upper()
        # Remove caracteres especiais mas mantém espaços e letras/números
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def create_features(self):
        """
        Cria matriz de características combinando múltiplas colunas relevantes
        """
        print("\n🔧 Criando matriz de características...")
        
        # Colunas mais relevantes para análise
        text_columns = [
            'Empenho (Histórico)(EOF)',
            'Função (Cod/Nome)(EOF)',
            'Subfunção (Cod/Nome)(EOF)',
            'Ação (Cod/Nome)(EOF)',
            'Programa (Cod/Nome)(EOF)',
            'Elemento Despesa (Cod/Nome)(EOF)',
            'Órgão (Código/Nome)(EOF)'
        ]
        
        # Combina textos de múltiplas colunas
        self.df['text_combined'] = ''
        for col in text_columns:
            if col in self.df.columns:
                self.df['text_combined'] += ' ' + self.df[col].fillna('').astype(str)
        
        # Pré-processa o texto combinado
        self.df['text_processed'] = self.df['text_combined'].apply(self.preprocess_text)
        
        # Vetorização TF-IDF
        print("  • Aplicando TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),  # Unigrams, bigrams e trigrams
            min_df=2,
            max_df=0.95,
            use_idf=True,
            sublinear_tf=True  # Aplica log(tf + 1)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(self.df['text_processed'])

        # Usa apenas características textuais (TF-IDF)
        self.features_matrix = tfidf_matrix.toarray()

        print(f"✓ Matriz de características criada: {self.features_matrix.shape}")
        print(f"  • Características baseadas apenas em texto (TF-IDF)")
        return self
    
    def cluster_dbscan(self):
        """
        Aplica DBSCAN para clustering com alta similaridade.
        Gera clusters para rotulação manual posterior.
        """
        print(f"\n🎯 Aplicando DBSCAN com vigilância ρ ≥ {self.vigilance}...")

        # Inicializa labels com -1 (não rotulado)
        self.labels = np.full(len(self.df), -1)
        self.confidence_scores = np.zeros(len(self.df))

        # Calcula distância epsilon baseada na vigilância
        # Vigilância de 0.9 significa similaridade mínima de 90%
        # Distância = 1 - similaridade
        eps = 1 - self.vigilance

        # DBSCAN clustering
        dbscan = DBSCAN(
            eps=eps,
            min_samples=2,  # Mínimo de 2 pontos para formar um cluster (mais granular)
            metric='cosine',  # Métrica de cosseno para dados textuais
            n_jobs=-1
        )

        cluster_labels = dbscan.fit_predict(self.features_matrix)

        # Estatísticas do clustering
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)

        print(f"  • Clusters encontrados: {n_clusters}")
        print(f"  • Pontos de ruído (não agrupados): {n_noise}")
        print(f"  • Clusters prontos para rotulação manual")

        # Atualiza labels com os clusters encontrados
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # Ignora ruído por enquanto
                cluster_mask = (cluster_labels == cluster_id)
                if cluster_mask.sum() > 0:
                    # Usa o cluster_id diretamente como label
                    self.labels[cluster_mask] = cluster_id
                    # Confiança inicial é zero, pois ainda não foram rotulados manualmente
                    self.confidence_scores[cluster_mask] = 0.0
        
        # Calcula métricas de qualidade do clustering
        if n_clusters > 1:
            valid_points = cluster_labels != -1
            if valid_points.sum() > 0:
                try:
                    silhouette = silhouette_score(
                        self.features_matrix[valid_points], 
                        cluster_labels[valid_points],
                        metric='cosine'
                    )
                    print(f"  • Coeficiente de Silhueta: {silhouette:.3f}")
                except:
                    pass
        
        return self
    
    def semi_supervised_learning(self, n_iterations=5):
        """
        Aplica aprendizado semi-supervisionado iterativo
        
        Args:
            n_iterations: Número de iterações
        """
        print(f"\n🤖 Iniciando aprendizado semi-supervisionado ({n_iterations} iterações)...")
        
        for iteration in range(n_iterations):
            print(f"\n  Iteração {iteration + 1}/{n_iterations}")
            
            # Separa dados rotulados e não rotulados
            labeled_mask = self.labels != -1
            n_labeled = labeled_mask.sum()
            n_unlabeled = (~labeled_mask).sum()
            
            print(f"    • Rotulados: {n_labeled}, Não rotulados: {n_unlabeled}")
            
            if n_labeled < 10 or n_unlabeled < 1:
                print("    ⚠ Poucos dados para continuar")
                break
            
            # Label Propagation
            label_prop = LabelPropagation(
                kernel='rbf',
                gamma=20,
                max_iter=1000
            )
            
            # Prepara dados para treinamento
            labels_train = self.labels.copy()
            
            # Treina o modelo
            label_prop.fit(self.features_matrix, labels_train)
            
            # Obtém probabilidades de predição
            proba_predictions = label_prop.predict_proba(self.features_matrix)
            
            # Atualiza labels com alta confiança
            confidence_threshold = 0.95 - (iteration * 0.05)  # Reduz threshold a cada iteração
            confidence_threshold = max(confidence_threshold, 0.75)  # Mínimo de 75%
            
            new_labels = 0
            for i in range(len(self.df)):
                if self.labels[i] == -1:  # Apenas não rotulados
                    max_proba = proba_predictions[i].max()
                    if max_proba >= confidence_threshold:
                        predicted_label = proba_predictions[i].argmax()
                        self.labels[i] = predicted_label
                        self.confidence_scores[i] = max_proba
                        new_labels += 1
            
            print(f"    • Novos rótulos atribuídos: {new_labels}")
            print(f"    • Threshold de confiança: {confidence_threshold:.2f}")
            
            # Salva histórico da iteração
            self.iteration_history.append({
                'iteration': iteration + 1,
                'n_labeled': n_labeled + new_labels,
                'n_unlabeled': n_unlabeled - new_labels,
                'new_labels': new_labels,
                'confidence_threshold': confidence_threshold
            })
            
            if new_labels == 0:
                print("    ⚠ Nenhum novo rótulo atribuído, parando iterações")
                break
        
        return self
    
    def analyze_results(self):
        """
        Analisa e visualiza os resultados da rotulação
        """
        print("\n📈 Analisando resultados...")
        
        # Adiciona labels ao dataframe
        self.df['label'] = self.labels
        self.df['confidence'] = self.confidence_scores
        self.df['label_name'] = self.df['label'].map(self.label_mapping).fillna('CLUSTER_' + self.df['label'].astype(str))
        
        # Estatísticas gerais
        print("\n📊 ESTATÍSTICAS GERAIS:")
        print(f"  • Total de registros: {len(self.df)}")
        print(f"  • Registros rotulados: {(self.labels != -1).sum()} ({(self.labels != -1).sum()/len(self.df)*100:.1f}%)")
        print(f"  • Registros não rotulados: {(self.labels == -1).sum()} ({(self.labels == -1).sum()/len(self.df)*100:.1f}%)")
        print(f"  • Confiança média: {self.confidence_scores[self.labels != -1].mean():.3f}")
        
        # Distribuição por label
        print("\n📊 DISTRIBUIÇÃO POR RÓTULO:")
        label_counts = self.df[self.df['label'] != -1]['label_name'].value_counts()
        for label, count in label_counts.head(20).items():
            print(f"  • {label}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Criar visualizações
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribuição de labels
        ax = axes[0, 0]
        top_labels = label_counts.head(15)
        ax.barh(range(len(top_labels)), top_labels.values)
        ax.set_yticks(range(len(top_labels)))
        ax.set_yticklabels(top_labels.index, fontsize=9)
        ax.set_xlabel('Quantidade')
        ax.set_title('Top 15 Rótulos Mais Frequentes')
        ax.grid(True, alpha=0.3)
        
        # 2. Evolução do aprendizado semi-supervisionado
        if self.iteration_history:
            ax = axes[0, 1]
            iterations = [h['iteration'] for h in self.iteration_history]
            labeled = [h['n_labeled'] for h in self.iteration_history]
            ax.plot(iterations, labeled, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Iteração')
            ax.set_ylabel('Registros Rotulados')
            ax.set_title('Evolução do Aprendizado Semi-Supervisionado')
            ax.grid(True, alpha=0.3)
        
        # 3. Distribuição de confiança
        ax = axes[0, 2]
        conf_data = self.confidence_scores[self.labels != -1]
        ax.hist(conf_data, bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(conf_data.mean(), color='red', linestyle='--', label=f'Média: {conf_data.mean():.2f}')
        ax.set_xlabel('Confiança')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição de Confiança dos Rótulos')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. PCA 2D dos clusters
        ax = axes[1, 0]
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(self.features_matrix)
        
        # Plot por label
        unique_labels = np.unique(self.labels[self.labels != -1])
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels[:20], colors):  # Limita a 20 labels para visualização
            mask = self.labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[color], alpha=0.6, s=10, 
                      label=self.label_mapping.get(label, f'Cluster {label}')[:20])
        
        # Pontos não rotulados
        mask_unlabeled = self.labels == -1
        ax.scatter(features_2d[mask_unlabeled, 0], features_2d[mask_unlabeled, 1],
                  c='gray', alpha=0.3, s=5, label='Não rotulado')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
        ax.set_title('Visualização PCA 2D dos Clusters')
        ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # 5. Matriz de valores por órgão
        ax = axes[1, 1]
        orgao_label = pd.crosstab(
            self.df['Órgão (Código/Nome)(EOF)'].str[:30],  # Trunca para caber
            self.df['label_name'].str[:20],
            values=self.df['Valor Empenhado (EOF)'],
            aggfunc='sum'
        ).fillna(0)
        
        # Top 10 órgãos por valor
        top_orgaos = orgao_label.sum(axis=1).nlargest(10).index
        top_labels_cols = orgao_label.sum(axis=0).nlargest(10).index
        
        subset = orgao_label.loc[top_orgaos, top_labels_cols]
        im = ax.imshow(subset.values, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(top_labels_cols)))
        ax.set_xticklabels(top_labels_cols, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(top_orgaos)))
        ax.set_yticklabels(top_orgaos, fontsize=8)
        ax.set_title('Heatmap: Valor Empenhado por Órgão e Rótulo')
        plt.colorbar(im, ax=ax)
        
        # 6. Temporal
        ax = axes[1, 2]
        self.df['mes'] = pd.to_datetime(self.df['Período (Dia/Mes/Ano)(EOF)']).dt.to_period('M')
        temporal = self.df.groupby(['mes', 'label_name'])['Valor Empenhado (EOF)'].sum().reset_index()
        
        for label in temporal['label_name'].unique()[:5]:  # Top 5 labels
            data = temporal[temporal['label_name'] == label]
            ax.plot(data['mes'].astype(str), data['Valor Empenhado (EOF)'], 
                   marker='o', label=label[:30])
        
        ax.set_xlabel('Período')
        ax.set_ylabel('Valor Empenhado (R$)')
        ax.set_title('Evolução Temporal dos Top 5 Rótulos')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('analise_rotulacao.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return self
    
    def export_results(self, output_path):
        """
        Exporta os resultados para arquivo Excel
        """
        print(f"\n💾 Exportando resultados para {output_path}...")
        
        # Prepara dataframe de saída
        output_df = self.original_df.copy()
        output_df['LABEL_ID'] = self.labels
        output_df['LABEL_NAME'] = self.df['label_name']
        output_df['CONFIDENCE_SCORE'] = self.confidence_scores
        output_df['LABELED'] = (self.labels != -1).astype(int)
        
        # Ordena por confiança e label
        output_df = output_df.sort_values(['LABELED', 'CONFIDENCE_SCORE'], ascending=[False, False])
        
        # Salva em Excel com múltiplas abas
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Aba principal com todos os dados
            output_df.to_excel(writer, sheet_name='Dados Rotulados', index=False)
            
            # Aba de estatísticas
            stats_df = pd.DataFrame([
                {'Métrica': 'Total de Registros', 'Valor': len(output_df)},
                {'Métrica': 'Registros Rotulados', 'Valor': (self.labels != -1).sum()},
                {'Métrica': 'Registros Não Rotulados', 'Valor': (self.labels == -1).sum()},
                {'Métrica': 'Percentual Rotulado', 'Valor': f"{(self.labels != -1).sum()/len(output_df)*100:.2f}%"},
                {'Métrica': 'Confiança Média', 'Valor': f"{self.confidence_scores[self.labels != -1].mean():.3f}"},
                {'Métrica': 'Total de Rótulos Únicos', 'Valor': len(np.unique(self.labels[self.labels != -1]))}
            ])
            stats_df.to_excel(writer, sheet_name='Estatísticas', index=False)
            
            # Aba com resumo por label
            summary_df = output_df[output_df['LABELED'] == 1].groupby('LABEL_NAME').agg({
                'LABEL_ID': 'count',
                'CONFIDENCE_SCORE': 'mean',
                'Valor Empenhado (EOF)': 'sum'
            }).round(2)
            summary_df.columns = ['Quantidade', 'Confiança Média', 'Valor Total Empenhado']
            summary_df = summary_df.sort_values('Quantidade', ascending=False)
            summary_df.to_excel(writer, sheet_name='Resumo por Rótulo')
            
            # Aba com histórico de iterações
            if self.iteration_history:
                history_df = pd.DataFrame(self.iteration_history)
                history_df.to_excel(writer, sheet_name='Histórico Iterações', index=False)
        
        print(f"✓ Resultados exportados com sucesso!")
        return output_path

# Execução principal
def main():
    """Função principal para executar o pipeline completo"""
    
    print("="*80)
    print("🚀 SISTEMA DE ROTULAÇÃO SEMI-SUPERVISIONADA DE DADOS ORÇAMENTÁRIOS")
    print("="*80)

    # Inicializa o sistema com vigilância de 0.9
    labeler = SemiSupervisedBudgetLabeler(vigilance=0.9)

    # Pipeline completo - Primeira etapa: Clustering
    (labeler
        .load_data('siof_saude.xlsx')
        .create_features()
        .cluster_dbscan()
        .analyze_results()
        .export_results('dados_clusters.xlsx')
    )

    print("\n" + "="*80)
    print("✅ CLUSTERING CONCLUÍDO COM SUCESSO!")
    print("="*80)

    # Recomendações finais
    print("\n📋 PRÓXIMOS PASSOS RECOMENDADOS:")
    print("1. Revise os clusters gerados na planilha 'dados_clusters.xlsx'")
    print("2. Rotule MANUALMENTE alguns exemplos de cada cluster principal")
    print("3. Salve os dados rotulados e execute o aprendizado semi-supervisionado")
    print("4. Use .semi_supervised_learning() para propagar os rótulos manuais")
    print("5. Ajuste o parâmetro de vigilância se necessário (atual: 0.9 = 90% similaridade)")
    
    return labeler

if __name__ == "__main__":
    labeler = main()
