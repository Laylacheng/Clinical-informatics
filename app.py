import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import os

# 1. 設定網頁頁面配置
st.set_page_config(page_title="生醫癌症基因臨床診斷面板", layout="wide")

st.title("🧬 互動式癌症基因分群與線上診斷模擬器")

# ==========================================
# 更新：實驗設計理念與資料集背景介紹區塊（加入 Kaggle 連結）
# ==========================================
with st.expander("📖 點擊展開：專案實驗設計理念與資料集背景介紹", expanded=True):
    st.markdown("""
    ### 🔬 1. 資料集來源與生醫背景
    本專案採用的生醫數據源自於 Kaggle 公開數據庫的 [ICMR Cancer Gene Expression Data](https://www.kaggle.com/datasets/shibumohapatra/icmr-data)。
    該資料集實質對接了著名的 **TCGA（The Cancer Genome Atlas，癌症基因體圖譜）** 數據庫。
    資料集內共包含 **801 筆病患樣本**，每筆樣本皆擁有 **20,530 個基因的 RNA-Seq 表現量數據（Gene Expression Data）**。
    這些樣本在臨床上已被證實分別屬於以下 **5 種不同的惡性腫瘤標籤（True Classes）**：
    * **BRCA (Breast Cancer)**：乳腺癌
    * **KIRC (Kidney Renal Cell Carcinoma)**：腎細胞癌
    * **COAD (Colon Adenocarcinoma)**：結腸癌
    * **LUAD (Lung Adenocarcinoma)**：肺腺癌
    * **PRAD (Prostate Adenocarcinoma)**：前列腺癌

    ### 🎯 2. 臨床痛點與實驗設計理念
    * **高維度資料的臨床挑戰**：在現代精準醫療中，RNA-Seq 技術能同時檢測上萬個基因的表現量，但這種**「超高維度（High-Dimensional）」**的資料特性，通常有嚴重的雜訊與特徵讓臨床醫師不直觀地進行疾病分類或診斷。
    * **PCA 技術**：本實驗的第一核心理念是利用 **主成分分析（PCA）** 演算法。PCA 能在不依賴任何臨床標籤的前提下，將 2 萬多維的基因特徵投影至 2 維或 3 維的關鍵主成分空間（PC Space），保留資料的變異量。
    * **K-Means 演算法的分群驗證**：**K-Means 機器學習分群演算法**（無監督學習），讓電腦純粹根據降維後的幾何距離對病患進行自動分群。
    * **臨床驗證核心（雙視角對比）**：本系統設計的重點在於**「演算法分群結果」與「臨床真實癌症標籤」的交叉比對**。若兩者的分布高度重合，即證明了利用 PCA 抓取的核心基因特徵，確實含有區分這五大癌症的關鍵生物學特徵。
    * **線上即時診斷模擬**：另外加上這個線上模擬器，能將新病人的高維度基因矩陣即時對齊、標準化、降維並指派群集，模擬 AI 輔助臨床癌症篩檢的決策流程。
    """)

st.markdown("---")

# ==========================================
# 檔案直接放在本機端目錄下
# ==========================================
DATA_PATH = "data.csv"
LABELS_PATH = "labels.csv"
TEST_FILE_PATH = "new_patients_test.csv"  # 範例測試檔案路徑

# 2. 智慧資料載入與本地快取機制
@st.cache_resource 
def load_large_biological_data():
    if not os.path.exists(DATA_PATH) or not os.path.exists(LABELS_PATH):
        st.error("🚨 錯誤：在本機端找不到 data.csv 或 labels.csv，請確認檔案已放置在與 app.py 相同的資料夾中。")
        st.stop()
        
    with st.spinner("正在讀取本機端生醫資料庫..."):
        df_data = pd.read_csv(DATA_PATH)
        df_labels = pd.read_csv(LABELS_PATH)
    return df_data, df_labels

# 執行載入與機器學習運算
try:
    df_data_raw, df_labels_raw = load_large_biological_data()
    
    # 3. 側邊欄參數設定 (Sidebar)
    st.sidebar.header("⚙️ 演算法參數調整")
    k_clusters = st.sidebar.slider("選擇 K-Means 分群數 (K)", min_value=2, max_value=10, value=5, step=1)
    
    # 4. 背景核心機器學習模型運算（快取處理）
    @st.cache_resource
    def train_models(df_data, df_labels, k_clusters):
        sample_ids = df_data.iloc[:, 0].copy()
        X = df_data.drop(["Unnamed: 0"], axis=1, errors='ignore')
        X = X.drop(["gene_5"], axis=1, errors='ignore')
        
        # 標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA 降維
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Sample_ID'] = sample_ids
        df_pca['Real_Class'] = df_labels['Class']
        
        # K-Means 分群
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init='auto')
        y_means = kmeans.fit_predict(X_pca)
        df_pca['Cluster_Result'] = y_means.astype(str)
        
        return scaler, pca, kmeans, df_pca, X.columns

    scaler, pca, kmeans, df_pca, feature_columns = train_models(df_data_raw, df_labels_raw, k_clusters)

    # 5. 網頁分頁配置
    tab1, tab2, tab3 = st.tabs(["📊 1. 資料概覽與手肘法", "🎨 2. PCA 互動式降維視覺化", "🔮 3. 線上即時診斷模擬器"])

    # --- 區塊 1 ---
    with tab1:
        st.subheader("📋 前處理後的基因表現量矩陣 (前五筆)")
        display_df = df_data_raw.drop(["Unnamed: 0", "gene_5"], axis=1, errors='ignore')
        st.dataframe(display_df.head(), use_container_width=True)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric(label="📊 總樣本數 (Patients)", value=display_df.shape[0])
        col_m2.metric(label="🧬 有效基因特徵數 (Genes)", value=display_df.shape[1])
        
        st.markdown("---")
        st.subheader("📈 手肘法 (Elbow Method) 最佳群數探索")
        
        @st.cache_data
        def calculate_wcss(X_pca_data):
            wcss_list = []
            for i in range(1, 11):
                km = KMeans(n_clusters=i, random_state=42, n_init='auto')
                km.fit(X_pca_data)
                wcss_list.append(km.inertia_)
            return wcss_list
        
        wcss_values = calculate_wcss(df_pca[['PC1', 'PC2', 'PC3']].values)
        df_wcss = pd.DataFrame({'群數 (K)': list(range(1, 11)), 'WCSS (群內平方和)': wcss_values})
        fig_wcss = px.line(df_wcss, x='群數 (K)', y='WCSS (群內平方和)', markers=True)
        st.plotly_chart(fig_wcss, use_container_width=True)

    # --- 區塊 2 ---
    with tab2:
        st.subheader("🎬 互動式生醫特徵空間投影")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            dimension_mode = st.radio("🔄 切換圖表維度：", ["2D 散佈圖", "3D 旋轉散佈圖"], horizontal=True)
        with col_ctrl2:
            color_view = st.selectbox("👁️ 切換觀看視角 (圖表著色依據)：", ["K-Means 演算法分群的結果", "病患真實的癌症類別"])
        
        color_column = 'Cluster_Result' if color_view == "K-Means 演算法分群的結果" else 'Real_Class'
        
        if dimension_mode == "2D 散佈圖":
            fig = px.scatter(
                df_pca, x='PC1', y='PC2', color=color_column,
                hover_data=['Sample_ID', 'Real_Class', 'Cluster_Result'],
                title=f"2D PCA 投影圖 (著色: {color_view})",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig.update_traces(marker=dict(size=8, opacity=0.8))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter_3d(
                df_pca, x='PC1', y='PC2', z='PC3', color=color_column,
                hover_data=['Sample_ID', 'Real_Class', 'Cluster_Result'],
                title=f"3D PCA 旋轉空間投影圖 (著色: {color_view})",
                color_discrete_sequence=px.colors.qualitative.Dark24
            )
            fig.update_traces(marker=dict(size=4, opacity=0.8))
            fig.update_layout(scene=dict(aspectmode='cube'))
            st.plotly_chart(fig, use_container_width=True)

    # --- 區塊 3 ---
    with tab3:
        st.subheader("🔮 臨床新樣本基因檢測與即時診斷")
        st.write("請下載下方提供的範例資料集，或上傳您自己的新病人基因表現量 CSV 檔案（需包含相同的基因特徵欄位）。")
        
        # ==========================================
        # 新增：提供給老師下載範例 CSV 的按鈕
        # ==========================================
        if os.path.exists(TEST_FILE_PATH):
            with open(TEST_FILE_PATH, "rb") as file:
                st.download_button(
                    label="📥 點我下載測試範例檔案 (new_patients_test.csv)",
                    data=file,
                    file_name="new_patients_test.csv",
                    mime="text/csv",
                    help="點擊下載包含 5 筆未知癌症病患的基因表現量矩陣，可用於下方即時診斷測試。"
                )
        else:
            st.info("💡 提示：未偵測到 new_patients_test.csv 範例檔，請確認檔案已同步至 GitHub 倉庫中。")
            
        st.markdown("---")
        
        uploaded_file = st.file_uploader("選擇上傳新病人基因數據 (.csv)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file)
                new_samples = df_new.iloc[:, 0].values
                X_new = df_new[feature_columns]
                
                X_new_scaled = scaler.transform(X_new)
                X_new_pca = pca.transform(X_new_scaled)
                preds = kmeans.predict(X_new_pca)
                
                df_res = pd.DataFrame({
                    '病人編號 (Sample ID)': new_samples,
                    'PCA - PC1': X_new_pca[:, 0].round(4),
                    'PCA - PC2': X_new_pca[:, 1].round(4),
                    '預測最接近群集群 (Predicted Cluster)': preds
                })
                
                st.markdown("---")
                st.subheader("🩺 診斷分析報告結果")
                st.dataframe(df_res, use_container_width=True)
                
                for idx, row in df_res.iterrows():
                    with st.expander(f"查看病人 {row['病人編號 (Sample ID)']} 詳細診斷建議"):
                        st.markdown(f"### 🎯 預測歸類：**Cluster {row['預測最接近群集群 (Predicted Cluster)']}**")
                        cluster_num = str(row['預測最接近群集群 (Predicted Cluster)'])
                        most_common_cancer = df_pca[df_pca['Cluster_Result'] == cluster_num]['Real_Class'].mode()[0]
                        st.warning(f"💡 臨床統計參考：在現有資料庫中，此群集主要聚集的真實癌症類型為 **{most_common_cancer}**。")
                        st.write("建議臨床醫師結合此基因分群特徵，安排進一步的臨床診斷。")
                        
            except Exception as e:
                st.error(f"❌ 解析錯誤：請確保上傳的檔案格式與欄位名稱完全正確。詳細錯誤: {str(e)}")

except Exception as e:
    st.error(f"🚨 錯誤：讀取本機端資料發生異常。詳細錯誤訊息: {str(e)}")
except Exception as e:
    st.error(f"🚨 錯誤：讀取本機端資料發生異常。詳細錯誤訊息: {str(e)}")
