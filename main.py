import os
import numpy as np
import pandas as pd
from src.data_preprocessing import *
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Carpeta donde está este script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Rutas seguras
    raw_data_path = os.path.join(current_dir, "data", "diabetes_dataset.csv")
    processed_data_path = os.path.join(current_dir, "data", "processed_diabetes.csv")

    # Mostrar información de depuración
    print("🔹 Buscando archivo CSV en la ruta:")
    print(raw_data_path)
    if not os.path.exists(raw_data_path):
        print("❌ ERROR: No se encontró el archivo CSV en la ruta indicada.")
        print("Por favor verifica que 'diabetes_dataset.csv' esté dentro de la carpeta 'data'.")
        return  # Salir del script si no encuentra el archivo
 
    # Si existe archivo procesado, cargarlo
    if os.path.exists(processed_data_path):
        print(" Archivo procesado encontrado. Cargando dataset existente...")
        df = pd.read_csv(processed_data_path)
    else:
        print(" Archivo procesado no encontrado. Ejecutando pipeline de preprocesamiento...")
        df = preprocesing_data(raw_data_path, processed_data_path)
        print(" Dataset procesado guardado en:", processed_data_path)

    print("Columnas en el DataFrame final:")
    print(df.columns.tolist())
 
    # Show preview of the data
    print("\nData preview:")
    print(df.head())

    print("\n📊 Iniciando análisis estadístico...")
    # Resumen estadístico general
    print("\nESTADÍSTICAS DESCRIPTIVAS DE VARIABLES NUMÉRICAS")
    print(df.describe())

    # Distribución de variables categóricas en porcentajes
    print("\nDISTRIBUCIÓN DE VARIABLES CATEGÓRICAS (%): ")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"{df[col].value_counts(normalize=True).round(3) * 100}\n")

    # Estadísticas adicionales: mediana,moda, rango y coeficiente de variación
    print("\nESTADÍSTICAS AVANZADAS")
    stats_extra = pd.DataFrame({
        'Mediana': df.median(numeric_only=True),
        'Moda': df.mode(numeric_only=True).iloc[0],
        'Rango': df.max(numeric_only=True) - df.min(numeric_only=True),
        'Coef_Variacion': (df.std(numeric_only=True) / df.mean(numeric_only=True)).round(3)
    })
    print(stats_extra)

    # Promedios por grupo de género o estado de diabetes
    print("\nPROMEDIOS DE IMC y DIABETES RISK SCORE POR GÉNERO")
    print(df.groupby('Gender')[['Body Mass Index', 'Diabetes Risk Score']].mean().round(2))

    print("\nPROMEDIOS DE EDAD, IMC Y DIABETES RISK SCORE POR ESTADO DE DIABETES")
    print(df.groupby('Diabetes Status')[['Age', 'Body Mass Index', 'Diabetes Risk Score']].mean().round(2))

    # Relación entre fumadores y Diabetes Risk
    print("\nCOMPARACIÓN ENTRE FUMADORES Y NO FUMADORES")
    mean_risk = df.groupby('Smoking')['Diabetes Risk Score'].agg(['mean', 'std', 'count']).round(2)
    mean_risk['error_std'] = (mean_risk['std'] / np.sqrt(mean_risk['count'])).round(2)
    print(mean_risk)

    # Correlación entre IMC y puntuación de riesgo
    print("\nCORRELACIÓN IMC vs RIESGO DE DIABETES ")
    corr_bmi_risk = df['Body Mass Index'].corr(df['Diabetes Risk Score'])
    print(f"Coeficiente de correlación: {corr_bmi_risk:.3f}")
    
    # Análisis de riesgo promedio por grupo de edad
    print("\nAGRUPACIÓN POR EDADES Y RIESGO DE DIABETES")
    df['Age Group'] = pd.cut(df['Age'], bins=[0,30,45,60,75,100],
                            labels=['<30','30-45','45-60','60-75','75+'])
    risk_by_age = df.groupby('Age Group', observed=True)['Diabetes Risk Score'].mean().round(2)
    print(risk_by_age)



    print("\nANÁLISIS DE ANTECEDENTES FAMILIARES MEDIANTE EL DIABETES RISK SCORE: ")
    # Mediana y Media del Riesgo Score por Antecedentes
    family_history_analysis = df.groupby('Family History')['Diabetes Risk Score'].agg(
        ['mean', 'median', 'std']
    )
    # (0=No Historial, 1=Con Historial)
    family_history_analysis.index = ['No Historial Familiar (0)', 'Con Historial Familiar (1)']
    print(family_history_analysis.round(2))
    # Cálculo de la Tasa de Riesgo Aumentada
    mean_risk_no_diabetes = family_history_analysis.loc['No Historial Familiar (0)', 'mean']
    mean_risk_with_diabetes = family_history_analysis.loc['Con Historial Familiar (1)', 'mean']
    risk_increase = ((mean_risk_with_diabetes - mean_risk_no_diabetes) / mean_risk_no_diabetes) * 100
    print(f"\nEl riesgo promedio de diabetes (nivel de amenaza) es un {risk_increase:.1f}% mayor en la población con antecedentes familiares.")


    # ACTIVIDAD FISICA Y RIESGO DE DIABETES
    print("\nRELACIÓN ENTRE RIESGO DE DIABETES Y ACTIVIDAD FÍSICA")
    df['Activity Level'] = pd.qcut(
    df['Physical Activity'], 
    q=3, 
    labels=['Baja', 'Media', 'Alta'], 
    duplicates='drop'
    )

    #CÁLCULO DE MÉTRICAS CLAVE POR NIVEL DE ACTIVIDAD
    activity_analysis = df.groupby('Activity Level', observed=True).agg(
        # Riesgo Promedio (Mean_Risk)
        Riesgo_Promedio=('Diabetes Risk Score', 'mean'),
        # IMC Promedio (Mean_IMC)
        IMC_Promedio=('Body Mass Index', 'mean'),
        # Prevalencia de Diabetes (calculando la media de la columna binaria 0/1)
        Prevalencia_Diabetes=('Diabetes Status', 'mean')
    )

    # Convierte la prevalencia a un porcentaje legible
    activity_analysis['Prevalencia_Diabetes (%)'] = (activity_analysis['Prevalencia_Diabetes'] * 100).round(1)
    # Redondea y presenta el resultado final
    activity_analysis = activity_analysis.drop(columns=['Prevalencia_Diabetes']).round(2)
    print(activity_analysis)
    # Análisis de la Brecha de Riesgo: Cuantificar cuánto peor es el grupo 'Baja' que el grupo 'Alta'
    risk_low = activity_analysis.loc['Baja', 'Riesgo_Promedio']
    risk_high = activity_analysis.loc['Alta', 'Riesgo_Promedio']
    risk_difference_percent = ((risk_low - risk_high) / risk_high) * 100
    print(f"\n *El grupo de Baja Actividad tiene un riesgo promedio de diabetes (nivel de amenaza) un {risk_difference_percent:.1f}% mayor que el de Alta Actividad.")
        

    print("\nPREVALENCIA DE DIABETES (%) POR ESTADO DE PESO (IMC)")
    # 1. Definir grupos de IMC
    bins_imc = [df['Body Mass Index'].min(), 25, 30, df['Body Mass Index'].max()]
    labels_imc = ['Bajo/Normal (<25)', 'Sobrepeso (25-30)', 'Obeso (>30)']
    df['IMC_Category'] = pd.cut(df['Body Mass Index'], bins=bins_imc, labels=labels_imc, right=False, include_lowest=True)
    # 2. Calcular la Prevalencia de Diabetes
    prevalence_analysis = df.groupby('IMC_Category', observed=True)['Diabetes Status'].mean().sort_values()
    # 3. Convertir a Porcentaje y Formatear
    prevalence_percent = (prevalence_analysis * 100).round(2)
    prevalence_percent.name = 'Prevalencia de Diabetes (%)'

    print(prevalence_percent)

    print("\n" + "="*80)
    print(" ANÁLISIS COMPARATIVO DE ESTILOS DE VIDA")
    print("="*80)

     # 1. ANÁLISIS DE ACTIVIDAD FÍSICA Y RIESGO DE DIABETES
    print("\n RIESGO DE DIABETES POR NIVEL DE ACTIVIDAD FÍSICA")
    df['Actividad_Fisica_Grupo'] = pd.cut(
        df['Physical Activity'],
        bins=[0, 60, 150, 300, 1000],
        labels=['Sedentario (<60min)', 'Bajo (60-150min)', 'Moderado (150-300min)', 'Alto (>300min)']
    )
    activity_risk = df.groupby('Actividad_Fisica_Grupo', observed=True)['Diabetes Risk Score'].agg(['mean', 'median', 'std', 'count'])
    print(activity_risk.round(2))


    # Diferencia entre sedentario y alto
    risk_sedentario = activity_risk.loc['Sedentario (<60min)', 'mean']
    risk_alto = activity_risk.loc['Alto (>300min)', 'mean']
    diff_percent = ((risk_sedentario - risk_alto) / risk_alto) * 100
    print(f"\n Insight: El grupo sedentario tiene un riesgo {diff_percent:.1f}% mayor que el grupo de alta actividad.")

 # 2. ANÁLISIS DE TIEMPO DE PANTALLA
    print("\nRIESGO DE DIABETES POR TIEMPO DE PANTALLA DIARIO")
    df['Screen_Time_Grupo'] = pd.cut(
        df['Screen Time'],
        bins=[0, 2, 4, 6, 24],
        labels=['Bajo (<2h)', 'Moderado (2-4h)', 'Alto (4-6h)', 'Muy Alto (>6h)']
    )
    screen_risk = df.groupby('Screen_Time_Grupo', observed=True)['Diabetes Risk Score'].agg(['mean', 'median', 'count'])
    print(screen_risk.round(2))
# Correlación pantalla-riesgo
    corr_screen = df['Screen Time'].corr(df['Diabetes Risk Score'])
    print(f"\n Correlación entre tiempo de pantalla y riesgo: {corr_screen:.3f}") 


 # 3. ANÁLISIS DE CALIDAD DE DIETA
    print("\n RIESGO DE DIABETES POR CALIDAD DE DIETA")
    df['Diet_Grupo'] = pd.cut(
        df['Diet'],
        bins=[0, 3, 6, 8, 10],
        labels=['Mala (0-3)', 'Regular (3-6)', 'Buena (6-8)', 'Excelente (8-10)']
    )
    diet_risk = df.groupby('Diet_Grupo', observed=True)['Diabetes Risk Score'].agg(['mean', 'median', 'count'])
    print(diet_risk.round(2))


  # 4. ANÁLISIS DE HORAS DE SUEÑO
    print("\n RIESGO DE DIABETES POR HORAS DE SUEÑO")
    df['Sleep_Grupo'] = pd.cut(
        df['Sleep Hours'],
        bins=[0, 6, 7, 8, 24],
        labels=['Poco (<6h)', 'Subóptimo (6-7h)', 'Óptimo (7-8h)', 'Excesivo (>8h)']
    )
    sleep_risk = df.groupby('Sleep_Grupo', observed=True)['Diabetes Risk Score'].agg(['mean', 'median', 'count'])
    print(sleep_risk.round(2))

    # 5. ANÁLISIS DE CONSUMO DE ALCOHOL
    print("\n RIESGO DE DIABETES POR CONSUMO DE ALCOHOL SEMANAL")
    df['Alcohol_Grupo'] = pd.cut(
        df['Alcohol per Week'],
        bins=[-1, 0, 5, 10, 100],
        labels=['Abstinente (0)', 'Bajo (1-5)', 'Moderado (6-10)', 'Alto (>10)']
    )
    alcohol_risk = df.groupby('Alcohol_Grupo', observed=True)['Diabetes Risk Score'].agg(['mean', 'median', 'count'])
    print(alcohol_risk.round(2))

 # 6. ANÁLISIS COMBINADO: MEJORES Y PEORES HÁBITOS
    print("\nCOMPARACIÓN: MEJORES vs PEORES HÁBITOS DE VIDA")
    
    # Definicion de grupos de hábitos saludables
    df['Habitos_Saludables'] = (
        (df['Physical Activity'] >= 150) &
        (df['Diet'] >= 6) &
        (df['Sleep Hours'] >= 7) & (df['Sleep Hours'] <= 8) &
        (df['Screen Time'] <= 4)
    ).astype(int)
    
    habitos_comparison = df.groupby('Habitos_Saludables')['Diabetes Risk Score'].agg(['mean', 'median', 'std', 'count'])
    habitos_comparison.index = ['Hábitos No Saludables', 'Hábitos Saludables']
    print(habitos_comparison.round(2))
    
    risk_diff = habitos_comparison.loc['Hábitos No Saludables', 'mean'] - habitos_comparison.loc['Hábitos Saludables', 'mean']
    print(f"\n Insight: Tener hábitos saludables reduce el riesgo en {risk_diff:.2f} puntos en promedio.")


#VISUALIZACIONES DE LOS GRAFICOS
    plot_path = os.path.join(current_dir, "data", "plots")
#CORELACION ENTRE IMC Y RIESGO DE DIABETES
    corr_bmi_risk = df['Body Mass Index'].corr(df['Diabetes Risk Score'])
    print(f"Coeficiente de correlación de Pearson: {corr_bmi_risk:.3f}")

    # Crear figura con estilo moderno
    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot con color por densidad
    scatter = ax.scatter(
        df['Body Mass Index'], 
        df['Diabetes Risk Score'],
        c=df['Diabetes Risk Score'],
        cmap='RdYlGn_r',  # Rojo=alto riesgo, Verde=bajo riesgo
        s=60,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.5
    )

    # Línea de regresión
    z = np.polyfit(df['Body Mass Index'], df['Diabetes Risk Score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['Body Mass Index'].min(), df['Body Mass Index'].max(), 100)
    ax.plot(x_line, p(x_line), 
            color='#e74c3c', 
            linewidth=3, 
            linestyle='--',
            label=f'Tendencia lineal')

    # Configuración de ejes
    ax.set_xlabel('Índice de Masa Corporal (IMC)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Puntuación de Riesgo de Diabetes', fontsize=14, fontweight='bold')
    ax.set_title(
        f'Correlación entre IMC y Riesgo de Diabetes\nPearson r = {corr_bmi_risk:.3f}',
        fontsize=16,
        fontweight='bold',
        pad=20
    )

    # Añadir líneas de referencia para categorías de IMC
    ax.axvline(x=25, color='orange', linestyle=':', linewidth=2, alpha=0.5, label='Sobrepeso (IMC=25)')
    ax.axvline(x=30, color='red', linestyle=':', linewidth=2, alpha=0.5, label='Obesidad (IMC=30)')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Nivel de Riesgo', fontsize=12, fontweight='bold')

    # Leyenda y grid
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Mejorar bordes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "CORELACION_ENTRE_IMC_Y_RIESGO_DE_DIABETES.png"))
    plt.show()


    
    

    #GRÁFICOS

    print("\n📊 MOSTRANDO GRÁFICOS...")

    # ===  DISTRIBUCIÓN DEL PUNTAJE DE RIESGO ===
    print("\nDistribución del Puntaje de Riesgo de Diabetes")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Distribución del Puntaje de Riesgo de Diabetes", fontsize=15, fontweight="bold")

    # 1. Distribución general del puntaje de riesgo
    sns.histplot(df["Diabetes Risk Score"], kde=True, bins=20, color="royalblue", ax=axs[0,0])
    axs[0,0].set_title("Distribución del Puntaje de Riesgo")
    axs[0,0].set_xlabel("Puntaje de Riesgo")
    axs[0,0].set_ylabel("Frecuencia")

    # 2. Puntaje de riesgo por grupo de edad
    sns.boxplot(data=df, x="Age Group", y="Diabetes Risk Score", palette="Set2", ax=axs[0,1])
    axs[0,1].set_title("Puntaje de Riesgo por Grupo de Edad")
    axs[0,1].set_xlabel("Grupo de Edad")
    axs[0,1].set_ylabel("Puntaje de Riesgo")
    axs[0,1].tick_params(axis='x', rotation=30)

    # 3. Riesgo de Diabetes vs HbA1c
    sns.scatterplot(data=df, x="Diabetes Risk Score", y="HbA1c", hue="Gender", alpha=0.7, ax=axs[1,0])
    sns.regplot(data=df, x="Diabetes Risk Score", y="HbA1c", scatter=False, color="red", ax=axs[1,0])
    axs[1,0].set_title("Riesgo de Diabetes vs HbA1c")
    axs[1,0].set_xlabel("Puntaje de Riesgo")
    axs[1,0].set_ylabel("HbA1c (%)")

    # 4. Actividad física vs Riesgo promedio
    sns.barplot(data=df, x="Activity Level", y="Diabetes Risk Score", estimator=np.mean, palette="viridis", ax=axs[1,1])
    axs[1,1].set_title("Riesgo Promedio por Nivel de Actividad")
    axs[1,1].set_xlabel("Nivel de Actividad Física")
    axs[1,1].set_ylabel("Riesgo Promedio")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plot_path, "Distribucion_del_Puntaje_de_Riesgo_de_Diabetes.png"))
    plt.show()

    print("✅ DISTRIBUCIÓN DEL PUNTAJE DE RIESGO completado: gráficos mostrados correctamente.")

    # === 👥FACTORES DEMOGRÁFICOS Y CONTROL DE DIABETES ===
    print("\n🔹Factores Demográficos y Control de Diabetes")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Factores Demográficos y Control de Diabetes", fontsize=15, fontweight="bold")

    # 1. HbA1c por nivel de ingreso
    sns.boxplot(data=df, x="Income Level", y="HbA1c", palette="coolwarm", ax=axs[0,0])
    axs[0,0].set_title("HbA1c por Nivel de Ingreso")
    axs[0,0].set_xlabel("Nivel de Ingreso")
    axs[0,0].set_ylabel("HbA1c (%)")
    axs[0,0].tick_params(axis='x', rotation=20)

    # 2. HbA1c promedio por grupo étnico
    sns.barplot(data=df, x="Ethnicity", y="HbA1c", estimator=np.mean, palette="mako", ax=axs[0,1])
    axs[0,1].set_title("Promedio de HbA1c por Etnia")
    axs[0,1].set_xlabel("Etnia")
    axs[0,1].set_ylabel("HbA1c Promedio (%)")
    axs[0,1].tick_params(axis='x', rotation=30)

    # 3. Relación Riesgo vs HbA1c
    sns.scatterplot(data=df, x="Diabetes Risk Score", y="HbA1c", hue="Income Level", alpha=0.7, ax=axs[1,0])
    sns.regplot(data=df, x="Diabetes Risk Score", y="HbA1c", scatter=False, color="red", ax=axs[1,0])
    axs[1,0].set_title("Riesgo vs HbA1c")
    axs[1,0].set_xlabel("Puntaje de Riesgo")
    axs[1,0].set_ylabel("HbA1c (%)")
    
    axs[1,1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plot_path, "Factores_Demograficos_y_Control_de_Diabetes.png"))
    plt.show()
    print("✅ FACTORES DEMOGRÁFICOS Y CONTROL DE DIABETES completado: gráficos mostrados correctamente.")


    # === MAPA DE CALOR DE CORRELACIÓN DE FACTORES CLÍNICOS ===
    print("\nMapa de Calor de Correlación de Factores Clínicos")
    clinical_cols = [
        'Body Mass Index',
        'Waist to Hip Ratio',
        'HbA1c',
        'Glucosa Ayunas',
        'Glucosa Postprandial',
        'Colesterol Total',
        'Colesterol HDL',
        'Colesterol LDL',
        'Presión Sistólica',
        'Presión Diastólica',
        'Diabetes Risk Score'
    ]

    existing_cols = [col for col in clinical_cols if col in df.columns]

    if not existing_cols:
        print("Ninguna columna clínica encontrada para el heatmap.")
    else:
        print("Columnas usadas para el mapa de calor:")
        print(existing_cols)

        df_heatmap = df[existing_cols].copy()
        # Convertir a numérico 
        for col in df_heatmap.columns:
            df_heatmap[col] = pd.to_numeric(df_heatmap[col], errors='coerce')

        corr_matrix = df_heatmap.corr()


        # Visualización del heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor='white',
            cbar_kws={"shrink": 0.8, "label": "Coeficiente de Correlación (r)"},
            annot_kws={"size": 9, "weight": "bold", "color": "black"}
        )

        plt.title(
            ' Mapa de Calor: Correlación entre Factores Clínicos',
            fontsize=15,
            fontweight='bold',
            pad=20
        )
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_path, "Correlación_entre_Factores_Clinicos.png"))
        plt.show()
            # === RIESGO DE DIABETES SEGÚN ESTILO DE VIDA ===
    print("\nVisualizando riesgo de diabetes según estilo de vida")

    plt.figure(figsize=(8,6))
    sns.barplot(
        data=df,
        x='Habitos_Saludables',
        y='Diabetes Risk Score',
        estimator=np.mean,
        palette=['#e74c3c', '#2ecc71']
    )

    plt.title('Riesgo de Padecer Diabetes según Estilo de Vida', fontsize=16, fontweight='bold')
    plt.xlabel('Estilo de Vida', fontsize=13)
    plt.ylabel('Puntuación Promedio de Riesgo de Diabetes', fontsize=13)
    plt.xticks(
        ticks=[0,1],
        labels=['Hábitos No Saludables', 'Hábitos Saludables'],
        fontsize=11
    )
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, "Riesgo_de_Padecer_Diabetes_segun_Estilo_de_Vida.png"))
    plt.show()

    # === ANÁLISIS DETALLADO: CADA HÁBITO Y SU RELACIÓN CON EL RIESGO ===
    print("\n Analizando cada hábito individual y su relación con el riesgo de diabetes...")

    #categorías para cada hábito
    df['Actividad_Fisica_Grupo'] = pd.cut(
        df['Physical Activity'],
        bins=[0, 60, 150, 300, 1000],
        labels=['Sedentario', 'Bajo', 'Moderado', 'Alto']
    )

    df['Dieta_Grupo'] = pd.cut(
        df['Diet'],
        bins=[0, 3, 6, 8, 10],
        labels=['Mala', 'Regular', 'Buena', 'Excelente']
    )

    df['Sueño_Grupo'] = pd.cut(
        df['Sleep Hours'],
        bins=[0, 6, 7, 8, 24],
        labels=['Poco (<6h)', 'Subóptimo (6-7h)', 'Óptimo (7-8h)', 'Excesivo (>8h)']
    )

    df['Alcohol_Grupo'] = pd.cut(
        df['Alcohol per Week'],
        bins=[-1, 0, 5, 10, 100],
        labels=['Abstinente (0)', 'Bajo (1-5)', 'Moderado (6-10)', 'Alto (>10)']
    )

    df['Pantalla_Grupo'] = pd.cut(
        df['Screen Time'],
        bins=[0, 2, 4, 6, 24],
        labels=['Bajo (<2h)', 'Moderado (2-4h)', 'Alto (4-6h)', 'Muy Alto (>6h)']
    )

    # Calcular el riesgo promedio por cada categoría de hábito
    riesgo_habitos = pd.DataFrame({
        'Actividad Física': df.groupby('Actividad_Fisica_Grupo')['Diabetes Risk Score'].mean().round(2),
        'Dieta': df.groupby('Dieta_Grupo')['Diabetes Risk Score'].mean().round(2),
        'Sueño': df.groupby('Sueño_Grupo')['Diabetes Risk Score'].mean().round(2),
        'Consumo de Alcohol': df.groupby('Alcohol_Grupo')['Diabetes Risk Score'].mean().round(2),
        'Tiempo de Pantalla': df.groupby('Pantalla_Grupo')['Diabetes Risk Score'].mean().round(2)
    })

    print("\nRiesgo promedio de diabetes por categoría de cada hábito:")
    print(riesgo_habitos)

        # === VISUALIZACIÓN COMPARATIVA  ===
    print("\n Visualizando relación entre hábitos y riesgo de diabetes ...")

    fig, axes = plt.subplots(2, 3, figsize=(16,10))
    fig.suptitle("Relación entre Estilo de Vida y Riesgo de Diabetes", fontsize=18, fontweight='bold')

    # Lista de configuraciones (columna, título, paleta, posición)
    habitos = [
        ('Actividad_Fisica_Grupo', 'Actividad Física', 'YlGnBu', axes[0,0]),
        ('Dieta_Grupo', 'Calidad de Dieta', 'YlOrBr', axes[0,1]),
        ('Sueño_Grupo', 'Horas de Sueño', 'PuBu', axes[0,2]),
        ('Alcohol_Grupo', 'Consumo de Alcohol', 'RdPu', axes[1,0]),
        ('Pantalla_Grupo', 'Tiempo de Pantalla', 'coolwarm', axes[1,1]),
    ]

    for col, titulo, paleta, ax in habitos:
        sns.barplot(
            data=df,
            x=col,
            y='Diabetes Risk Score',
            estimator=np.mean,
            palette=paleta,
            ax=ax
        )

        # Etiquetas y formato
        ax.set_title(titulo, fontsize=13, fontweight='bold')
        ax.set_xlabel("")
        ax.set_ylabel("Riesgo Promedio" if "Actividad" in titulo or "Alcohol" in titulo else "")
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        # === Escala automática ===
        mean_vals = df.groupby(col)['Diabetes Risk Score'].mean()
        ax.set_ylim(mean_vals.min() - 1, mean_vals.max() + 1)

        # === Etiquetas numéricas ===
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=9, padding=3)

    # Último subplot vacío
    axes[1,2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(plot_path, "Relacion_entre_Estilo_de_Vida_y_Riesgo_de_Diabetes.png"))
    plt.show()

if __name__ == "__main__":
    main()