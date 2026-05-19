import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analizador import DataAnalyzer

st.set_page_config(page_title="Proyecto Python for Analytics - A. Vargas", layout="wide")

st.sidebar.title("Estructura del Proyecto")
opcion = st.sidebar.selectbox("Selecciona un módulo:", ["Módulo 1: Home", "Módulo 2: EDA"], key="mod_sel")

if opcion == "Módulo 1: Home":
    st.title("Caso de Estudio N°2: Retención y Fuga de Clientes (Churn)")
    st.subheader("Presentación y Contexto del Negocio")
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Datos del Proyecto
        * **Estudiante:** Anthony Vargas Aquino
        * **Programa:** Especialización en Python for Analytics
        * **Año:** 2026
        
        ### Contexto del Negocio
        Evaluamos el comportamiento de la cartera de clientes utilizando el dataset *Telco Customer Churn*. El objetivo central de este desarrollo es identificar los perfiles críticos de usuarios y aislar los factores comerciales, contractuales y técnicos que están acelerando la deserción dentro de la compañía para diseñar estrategias de retención de la mano con el negocio.
        """)
    with col2:
        try:
            st.image("logo_dmc.png", use_container_width=True)
        except:
            pass

elif opcion == "Módulo 2: EDA":
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    @st.cache_data
    def cargar_datos():
        df = pd.read_csv("TelcoCustomerChurn.csv")
        df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        return df
        
    try:
        df_raw = cargar_datos()
        df_clean = df_raw.copy().drop(columns=['customerID'], errors='ignore')
        st.session_state['df_churn'] = df_clean
        analizador = DataAnalyzer(df_clean)
        
        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs([
            "Info", "Clasificación", "Estadísticas", "Nulos", "Numéricas", 
            "Categóricas", "Bivariado (N-C)", "Bivariado (C-C)", "Filtros", "Hallazgos", "Conclusiones"
        ])

        with t1:
            st.subheader("Información general")
            c1, c2 = st.columns(2)
            c1.dataframe(df_raw.dtypes.astype(str).to_frame(name="Tipo"), use_container_width=True)
            c2.dataframe(df_raw.isnull().sum().to_frame(name="Nulos"), use_container_width=True)

        with t2:
            st.subheader("Clasificación de variables")
            num, cat = analizador.clasificar_variables()
            c1, c2 = st.columns(2)
            c1.markdown(f"**Variables Numéricas ({len(num)}):**")
            c1.dataframe(pd.DataFrame(num, columns=["Columnas"]), use_container_width=True)
            c2.markdown(f"**Variables Categóricas ({len(cat)}):**")
            c2.dataframe(pd.DataFrame(cat, columns=["Columnas"]), use_container_width=True)

        with t3:
            st.subheader("Estadísticas descriptivas")
            st.dataframe(analizador.obtener_descriptivas(), use_container_width=True)
            st.markdown("""
            **Notas de análisis:**
            * **Tenure:** Registro una dispersión importante (std de 24.5). La cercanía entre la media (32) and la mediana (29) me indica que el comportamiento de la antigüedad está relativamente balanceado en la muestra general.
            * **MonthlyCharges:** Observo que la mediana (70) supera a la media (64), lo que me confirma una fuerte inclinación de la base de usuarios hacia la contratación de planes comerciales premium.
            * **TotalCharges:** La brecha entre la media y la mediana se explica por el peso que ejercen los clientes cautivos de larga permanencia en la facturación acumulada.
            """)

        with t4:
            st.subheader("Análisis de valores nulos")
            c_tabla, c_grafico = st.columns([3, 2])
            with c_tabla:
                df_nulos = analizador.analizar_nulos()
                st.dataframe(df_nulos, use_container_width=True)
            with c_grafico:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                df_nulos_filtrados = df_nulos[df_nulos['Conteo'] > 0]
                if not df_nulos_filtrados.empty:
                    df_nulos_filtrados['Conteo'].plot(kind='bar', ax=ax, color='salmon', width=0.2)
                    ax.set_ylabel("Frecuencia")
                    plt.xticks(rotation=45, fontsize=8)
                    plt.yticks(fontsize=8)
                    st.pyplot(fig)
                else:
                    st.info("No se registran valores faltantes en las variables.")
            st.markdown("""
            **Comentario:** Identifico un volumen marginal de nulos concentrado exclusivamente en la columna TotalCharges (11 registros). Al representar menos del 0.05% de la base total, procedo con el EDA sabiendo que su peso estadístico no distorsionará las métricas financieras globales.
            """)

        with t5:
            st.subheader("Distribución de variables numéricas")
            df_plot = df_clean.dropna(subset=['TotalCharges'])
            analizador_plot = DataAnalyzer(df_plot)
            col = st.selectbox("Variable:", df_plot.select_dtypes(include=['float64', 'int64']).columns, key="num_dist_sel")
            st.pyplot(analizador_plot.plot_distribucion(col, 'hist'))
            st.markdown("""
            **Lectura de distribución:**
            * Evalúo la forma de la curva de densidad para mapear acumulaciones críticas de datos, detectar asimetrías y entender la concentración de frecuencias en la variable seleccionada.
            """)

        with t6:
            st.subheader("Distribución de variables categóricas")
            col = st.selectbox("Categoría:", [c for c in df_clean.columns if df_clean[c].dtype == 'object' or df_clean[c].dtype.name == 'category'], key="cat_dist_sel")
            c1, c2 = st.columns([2, 3]) 
            with c1:
                st.dataframe(analizador.analizar_categoricas(col), use_container_width=True)
            with c2:
                st.pyplot(analizador.plot_barras(col), use_container_width=False)

        with t7:
            num_var, cat_var = analizador.clasificar_variables()
            num = st.selectbox("Numérica:", num_var, key="biv_num_sel")
            cat = st.selectbox("Categoría:", cat_var, key="biv_cat_sel")
            c_izq, c_centro, c_der = st.columns([1, 2, 1])
            with c_centro:
                st.pyplot(analizador.plot_bivariado_num_cat(num, cat), use_container_width=True)

        with t8:
            st.subheader("Relación entre variables categóricas")
            v1 = st.selectbox("V1:", [c for c in df_clean.columns if df_clean[c].dtype == 'object' or df_clean[c].dtype.name == 'category'], key="cat_cat_1")
            v2 = st.selectbox("V2:", [c for c in df_clean.columns if df_clean[c].dtype == 'object' or df_clean[c].dtype.name == 'category'], key="cat_cat_2")
            fig, tabla = analizador.plot_bivariado_cat_cat(v1, v2)
            c_izq, c_centro, c_der = st.columns([1, 3, 1])
            with c_centro:
                st.markdown("##### Tabla de contingencia")
                st.dataframe(tabla, use_container_width=True)
                st.markdown("##### Visualización bivariada apilada")
                st.pyplot(fig, use_container_width=True)

        with t9:
            st.subheader("Análisis dinámico según columnas elegidas")
            cols = st.multiselect("Selecciona las columnas para analizar:", df_clean.columns.tolist(), default=df_clean.columns.tolist()[:3], key="dyn_sel")
            if cols:
                st.markdown("##### Vista previa de los datos seleccionados")
                st.dataframe(df_clean[cols].head(10), use_container_width=True)
                df_sub = df_clean[cols]
                num_sub = df_sub.select_dtypes(include=['float64', 'int64']).columns.tolist()
                cat_sub = [c for c in df_sub.columns if c not in num_sub]
                c1, c2 = st.columns(2)
                with c1:
                    if num_sub:
                        st.markdown("##### Resumen estadístico numérico")
                        st.dataframe(df_sub[num_sub].describe(), use_container_width=True)
                    else:
                        st.info("Selecciona al menos una variable cuantitativa.")
                with c2:
                    if cat_sub:
                        st.markdown("##### Detalle de valores únicos (Variables categóricas)")
                        lista_categorias = []
                        for c in cat_sub:
                            valores_unicos = ", ".join(df_sub[c].dropna().unique().astype(str))
                            lista_categorias.append({"Variable Categórica": c, "Valores Únicos": valores_unicos})
                        df_resumen_cat = pd.DataFrame(lista_categorias).set_index("Variable Categórica")
                        st.dataframe(df_resumen_cat, use_container_width=True)
                    else:
                        st.info("Selecciona al menos una variable cualitativa.")
            else:
                st.warning("Selecciona parámetros en el menú superior para desplegar las métricas interactivas.")

        with t10:
            st.subheader("Ítem 10: Hallazgos clave de las relaciones bivariadas")
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            sns.violinplot(data=df_clean, x='Churn', y='MonthlyCharges', ax=axes[0, 0], palette="muted")
            axes[0, 0].set_title("Cargos Mensuales por Estado de Fuga", fontsize=9, fontweight='bold')
            sns.violinplot(data=df_clean, x='Churn', y='tenure', ax=axes[0, 1], palette="muted")
            axes[0, 1].set_title("Antigüedad (Tenure) por Estado de Fuga", fontsize=9, fontweight='bold')
            ct_contract = pd.crosstab(df_clean['Contract'], df_clean['Churn'])
            ct_contract.plot(kind='bar', stacked=True, ax=axes[1, 0], color=['#440154', '#fde725'])
            axes[1, 0].set_title("Fuga según Tipo de Contrato", fontsize=9, fontweight='bold')
            ct_internet = pd.crosstab(df_clean['InternetService'], df_clean['Churn'])
            ct_internet.plot(kind='bar', stacked=True, ax=axes[1, 1], color=['#440154', '#fde725'])
            axes[1, 1].set_title("Fuga según Servicio de Internet", fontsize=9, fontweight='bold')
            plt.tight_layout()
            c_izq, c_centro, c_der = st.columns([1, 3, 1])
            with c_centro:
                st.pyplot(fig, use_container_width=True)
            st.markdown("---")
            col_insights_1, col_insights_2 = st.columns(2)
            with col_insights_1:
                st.markdown("""
                ### Relación Financiera y Temporal
                * **Impacto de los Cargos Mensuales (MonthlyCharges):** Detecto que la masa crítica de deserción (Churn = Yes) se concentra de forma muy marcada en las tarifas que oscilan entre 70 y 105. Por el contrario, los clientes retenidos muestran su mayor densidad en el escalón básico de $20, lo que expone que los costos mensuales altos están actuando como un gatillo de salida directo.
                * **Efecto de la Antigüedad (tenure):** La forma del violín me evidencia que el abandono es un problema de adopción temprana, acumulándose con fuerza en los primeros 5 meses de la cuenta. Si logro estabilizar al cliente durante el año inicial, la curva de deserción cae y el perfil se fideliza a largo plazo.
                """)
            with col_insights_2:
                st.markdown("""
                ### Influencia del Servicio
                * **Vulnerabilidad Contractual (Contract):** El esquema comercial mes a mes (Month-to-month) absorbe casi el total de las cancelaciones del negocio. Las barras de contratos comerciales fijos (de 1 y 2 años) retienen de manera mucho más firme el flujo operativo de la cartera.
                * **Riesgo en Infraestructura (InternetService):** Aislando el tipo de red, detecto que los usuarios con conexión de fibra óptica registran proporciones de deserción muy elevadas. Esto me obliga a poner bajo la lupa la estabilidad del servicio técnico o el nivel de precios de este producto específico, ya que la infraestructura de mayor velocidad está fallando en retener.
                """)

        with t11:
            st.subheader("Conclusiones Finales y Plan de Acción Comercial")
            st.markdown("""
            1. **La modalidad contractual determina el riesgo inmediato:** El contrato mes a mes (Month-to-month) es la principal vía de fuga operativa del negocio. **Estrategia comercial:** Recomiendo estructurar un plan de migración forzada hacia contratos anuales mínimos, absorbiendo parte del costo de instalación o agregando bonificaciones de velocidad en red para blindar la base recurrente.
            
            2. **Fuga prematura concentrada en el ciclo inicial:** El análisis de la variable temporal me demuestra que el abandono se focaliza críticamente durante el primer semestre de uso. **Estrategia comercial:** Es clave implementar un protocolo prioritario de servicio al cliente durante los primeros 180 días de la cuenta, ejecutando llamadas preventivas de soporte e identificando fricciones antes de que el usuario decida rescindir el servicio.
            
            3. **La Fibra Óptica presenta una alta fricción comercial o técnica:** Aunque constituye el producto premium de mayor velocidad, este segmento lidera la tasa de bajas. **Estrategia comercial:** Planteo auditar de forma urgente los indicadores de estabilidad en la infraestructura técnica de fibra y cruzar estos datos con encuestas directas para verificar si el descontento obedece a fallas de conexión o a una percepción desequilibrada de costo-beneficio.
            
            4. **Las tarifas elevadas actúan como un disparador de deserción:** La concentración de bajas se dispara sensiblemente por encima de los $70 mensuales. **Estrategia comercial:** Se requiere flexibilizar las estructuras de precios del catálogo actual a través de la creación de paquetes (bundles) modulares que diluyan el costo individual de las suscripciones, incrementando el valor percibido por el cliente.
            
            5. **Calidad de la base de información estructural:** El inventario de datos arrojó un estado de integridad impecable, aislando apenas 11 casos con nulos en la facturación total. **Estrategia comercial:** Esta pureza en el set de datos me permite validar las conclusiones alcanzadas con un nivel de confianza absoluto, asegurando que las decisiones comerciales ejecutadas están basadas en el comportamiento real y completo del negocio.
            """)

    except Exception as e:
        st.error(f"Error en la ejecución: {e}")