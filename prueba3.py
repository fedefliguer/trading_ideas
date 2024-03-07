import time
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

start_time = time.time()

def calculo_resistencias_historia(ticker, df, lags, rango_quiebre, arc, Logging):
    """
    Determina para el dataset con los precios, los valores de las resistencias que el precio tiene y que alguna vez tuvo pero fueron quebradas.

    Recibe: 
        ticker: string con el código del ticker.
        df: dataset con la información del activo en el tiempo.
        lags: días entre los cuales tiene que ser un máximo para ser considerada una resistencia.
        rango_quiebre: porcentaje que debe pasarse para que la resistencia se considere vencida.
        arc: armar resistencias con. Puede ser "Close" o "High" si se quiere armar en intradiario.
        logging: True para ir viendo en el backtesting cada una de las etapas y poder detectar errores.

    Devuelve: 
        df_all_fijas: listado total de resistencias vigentes a la fecha, con la información de las pruebas que tuvo.
        df_all_quebradas: listado total de resistencias ya quebradas, con la información de las pruebas que tuvo como soporte.
    """
    i = 1
    while i < (lags + 1):  # Genera ventanas alrededor para analizar si es resistencia
        colname = "h%sb" % (i)
        df[colname] = round(df[arc].shift(i), 2)
        colname = "h%sf" % (i)
        df[colname] = round(df[arc].shift(-i), 2)
        i = i + 1
    x = np.arange(len(df.Date))
    df["maxb"] = round(df.filter(regex=("h.*b")).max(axis=1), 2)
    df["maxf"] = round(df.filter(regex=("h.*f")).max(axis=1), 2)
    df["Resistencia"] = np.where(
        (df.index > lags) & (df[arc] > df["maxb"]) & (df[arc] > df["maxf"]), 1, 0
    )
    df["Resistencia"] = np.where(
        (df.index > (df.shape[0] - lags)), 0, df.Resistencia * df[arc]
    )
    df = df[["Date", "Ticker", "Close", "High", "Low", "Resistencia"]].copy()
    df.loc[:, "Date_vigencia"] = df["Date"].shift(-lags)
    df.replace(0, np.nan, inplace=True)
    puntos_resistencia = df[df.Resistencia > 0]["Resistencia"].to_numpy()
    fechas_resistencia = df[df.Resistencia > 0][
        "Date"
    ].to_numpy()  # Lista de fechas de soporte
    fechas_vigencia_resistencia = df[df.Resistencia > 0][
        "Date_vigencia"
    ].to_numpy()  # Lista de fechas de soporte
    df_all_fijas = pd.DataFrame()
    df_all_quebradas = pd.DataFrame()

    i = 0
    print("\n", "Análisis de resistencias fijas: ") if Logging else None
    while i < len(puntos_resistencia):  # Recorre la lista de resistencias
        print(
            "Analizo el punto de resistencia ",
            round(puntos_resistencia[i], 2),
            " del día ",
            fechas_resistencia[i],
            " que entra en vigencia el día ",
            fechas_vigencia_resistencia[i],
        ) if Logging else None
        df_all_fijas = recorrido_resistencia(
            ticker,
            df,
            fechas_vigencia_resistencia[i],
            puntos_resistencia[i],
            rango_quiebre,
            df_all_fijas,
            1,
            Logging,
        )
        i = i + 1

    i = 0
    j = 0
    print("\n", "Análisis de resistencias fijas quebradas: ") if Logging else None
    while i < len(puntos_resistencia):  # Recorre la lista de resistencias
        print(
            "Analizo el punto de resistencia ",
            round(puntos_resistencia[i], 2),
            " del día ",
            fechas_resistencia[i],
            " que entra en vigencia el día ",
            fechas_vigencia_resistencia[i],
        ) if Logging else None
        mascara_resistencia_quebrada = (df.Date > fechas_vigencia_resistencia[i]) & (
            df[arc] > (puntos_resistencia[i] * (1 + rango_quiebre))
        )
        nunca_fue_quebrado = len(df[mascara_resistencia_quebrada]) == 0
        if nunca_fue_quebrado:
            print(" Nunca fue quebrado hacia arriba") if Logging else None
        else:
            fecha_quiebre_resistencia = df[mascara_resistencia_quebrada].Date.iloc[0]
            print(
                " Fue quebrado hacia arriba el día ",
                fecha_quiebre_resistencia,
                ". Lo analizo como soporte",
            ) if Logging else None
            df_all_quebradas = recorrido_soporte(
                ticker,
                df,
                fecha_quiebre_resistencia,
                puntos_resistencia[i],
                rango_quiebre,
                df_all_quebradas,
                1,
                Logging,
            )
        i = i + 1
    return df_all_fijas, df_all_quebradas

def calculo_soportes_historia(ticker, df, lags, rango_quiebre, asc, Logging):
    """
    Determina para el dataset con los precios, los valores de los soportes que el precio tiene y que alguna vez tuvo pero fueron quebrados.

    Recibe: 
        ticker: string con el código del ticker.
        df: dataset con la información del activo en el tiempo.
        lags: días entre los cuales tiene que ser un mínimo para ser considerado un soporte.
        rango_quiebre: porcentaje que debe pasarse para que el soporte se considere quebrado.
        asc: armar soportes con. Puede ser "Close" o "High" si se quiere armar en intradiario.
        logging: True para ir viendo en el backtesting cada una de las etapas y poder detectar errores.

    Devuelve: 
        df_all_fijas: listado total de soportes vigentes a la fecha, con la información de las pruebas que tuvo.
        df_all_quebradas: listado total de soportes ya quebradas, con la información de las pruebas que tuvo como resistencia.
    """

    i = 1
    while i < (lags + 1):  # Genera ventanas alrededor para analizar si es soporte
        colname = "l%sb" % (i)
        df[colname] = round(df[asc].shift(i), 2)
        colname = "l%sf" % (i)
        df[colname] = round(df[asc].shift(-i), 2)
        i = i + 1
    x = np.arange(len(df.Date))
    df["minb"] = round(df.filter(regex=("l.*b")).min(axis=1), 2)
    df["minf"] = round(df.filter(regex=("l.*f")).min(axis=1), 2)
    
    df["Soporte"] = np.where(
        (df.index > lags) & (df[asc] < df["minb"]) & (df[asc] < df["minf"]), 1, 0
    )
    df["Soporte"] = np.where((df.index > (df.shape[0] - lags)), 0, df.Soporte * df[asc])
    df = df[["Date", "Ticker", "Close", "Low", "High", "Soporte"]].copy()
    df.replace(0, np.nan, inplace=True)
    puntos_soporte = df[df.Soporte > 0]["Soporte"].to_numpy()  # Lista de soportes
    df.loc[:, "Date_vigencia"] = df["Date"].shift(-lags)
    
    fechas_soporte = df[df.Soporte > 0]["Date"].to_numpy()  # Lista de fechas de soporte
    fechas_vigencia_soporte = df[df.Soporte > 0][
        "Date_vigencia"
    ].to_numpy()  # Lista de fechas de soporte
    df_all_fijas = pd.DataFrame()
    df_all_quebradas = pd.DataFrame()
    i = 0
    print("\n", "Análisis de soportes fijos: ") if Logging else None
    while i < len(puntos_soporte):  # Recorre la lista de soportes
        print(
            "Analizo el punto de soporte ",
            round(puntos_soporte[i], 2),
            " del día ",
            fechas_soporte[i],
            " que entraría en vigencia el día ",
            fechas_vigencia_soporte[i],
        ) if Logging else None
        df_all_fijas = recorrido_soporte(
            ticker,
            df,
            fechas_vigencia_soporte[i],
            puntos_soporte[i],
            rango_quiebre,
            df_all_fijas,
            1,
            Logging,
        )
        i = i + 1

    i = 0
    j = 0
    print("\n", "Análisis de soportes fijos quebrados: ") if Logging else None
    while i < len(puntos_soporte):  # Recorre la lista de resistencias
        print(
            "Analizo el punto de soporte ",
            round(puntos_soporte[i], 2),
            " del día ",
            fechas_soporte[i],
            " que entra en vigencia el día ",
            fechas_vigencia_soporte[i],
        ) if Logging else None
        mascara_soporte_quebrado = (df.Date > fechas_vigencia_soporte[i]) & (
            df[asc] > (puntos_soporte[i] * (1 + rango_quiebre))
        )
        nunca_fue_quebrado = len(df[mascara_soporte_quebrado]) == 0
        if nunca_fue_quebrado:
            print(" Nunca fue quebrado hacia abajo") if Logging else None
        else:
            fecha_quiebre_soporte = df[mascara_soporte_quebrado].Date.iloc[0]
            print(
                " Fue quebrado hacia abajo el día ",
                fecha_quiebre_soporte,
                ". Lo analizo como resistencia",
            ) if Logging else None
            df_all_quebradas = recorrido_resistencia(
                ticker,
                df,
                fecha_quiebre_soporte,
                puntos_soporte[i],
                rango_quiebre,
                df_all_quebradas,
                1,
                Logging,
            )
        i = i + 1
    return df_all_fijas, df_all_quebradas

def recorrido_resistencia(ticker, dataset, fecha_empieza_vigencia, valor_resistencia, rango_quebrado, df_all, prueba_nro, Logging,):
    """
    Para una resistencia, evalua su performance después (si fue probada o si fue quebrada)

    Recibe: 
        ticker: string con el código del ticker.
        dataset: dataset con la información del activo en el tiempo.
        fecha_empieza_vigencia: día en el que la resistencia entra en vigor para ver si pudo ser probada.
        valor_resistencia: precio de resistencia.
        rango_quebrado: rango sobre el cual se dice que la resistencia fue quebrada.
        df_all: dataset donde se guarda.
        prueba_nro: número de vez en la que la misma resistencia es probada
        logging: True para ir viendo en el backtesting cada una de las etapas y poder detectar errores.

    Devuelve: 
        df_all: dataset que guarda todo el registro de las pruebas de los soportes y las resistencias.
    """

    dataset = dataset[dataset.Date > fecha_empieza_vigencia].copy()
    dataset["rango_quebrado_inicia"] = valor_resistencia * (1 - rango_quebrado)
    dataset["valor"] = valor_resistencia
    dataset["rango_quebrado_termina"] = valor_resistencia * (1 + rango_quebrado)
    dataset["es_zona_prueba"] = np.where(
        (dataset["Low"] < dataset["valor"]) & (dataset["High"] > dataset["valor"]), 1, 0
    )
    dataset["es_zona_quiebre"] = np.where(
        (dataset["High"] > dataset["rango_quebrado_termina"]), 1, 0
    )
    dataset["es_zona_confirmacion"] = np.where(
        (dataset["Low"] < dataset["rango_quebrado_inicia"]), 1, 0
    )
    mascara_probado = dataset.es_zona_prueba == 1
    mascara_quebrado = dataset.es_zona_quiebre == 1
    nunca_fue_probado_ni_quebrado = (len(dataset[mascara_probado]) == 0) & (len(dataset[mascara_quebrado]) == 0)
    nunca_fue_probado = len(dataset[mascara_probado]) == 0
    fue_quebrado = len(dataset[mascara_quebrado]) > 0
    if nunca_fue_probado_ni_quebrado:  # NO FUE PROBADO NI QUEBRADO
        resolucion = "vigente"
        fecha_prueba = np.nan
        fecha_resolucion = np.nan
        print(
            "La prueba número ", prueba_nro, "nunca se realizó en la historia"
        ) if Logging else None
    elif nunca_fue_probado:  # QUEBRADO SIN PROBARSE
        resolucion = "quebrado"
        fecha_prueba = dataset[mascara_quebrado].iloc[0].Date
        fecha_resolucion = dataset[mascara_quebrado].iloc[0].Date
        print(
            "La prueba",
            prueba_nro,
            "nunca se realizó, pero la resistencia fue quebrada",
        ) if Logging else None
    else:
        fecha_prueba = dataset[mascara_probado].iloc[0].Date
        resolucion = ""
        if fue_quebrado:
            fecha_quiebre = dataset[mascara_quebrado].iloc[0].Date
            if fecha_quiebre < fecha_prueba:
                resolucion = "quebrado"
                fecha_prueba = np.nan
                fecha_resolucion = fecha_quiebre
                print(
                    "La prueba", prueba_nro, "fue quebrada antes de probarse"
                ) if Logging else None
        if resolucion != "quebrado":
            print(
                "La prueba número ",
                prueba_nro,
                "empezó el día ",
                fecha_prueba,
                " cuando el mínimo fue ",
                dataset[mascara_probado].iloc[0].Low,
                " y el máximo fue ",
                dataset[mascara_probado].iloc[0].High,
            ) if Logging else None
            mascara_resuelto = (
                (dataset.Date > fecha_prueba) & (dataset.es_zona_confirmacion == 1)
            ) | ((dataset.Date > fecha_prueba) & (dataset.es_zona_quiebre == 1))
            fue_resuelta = len(dataset[mascara_resuelto]) > 0
            if fue_resuelta:  # La prueba fue resuelta
                primera_resolucion = dataset[mascara_resuelto].iloc[0]
                fecha_resolucion = primera_resolucion.Date
                fue_resuelta_indeterminada = (primera_resolucion.es_zona_confirmacion == 1) & (primera_resolucion.es_zona_quiebre == 1)
                fue_resuelta_probada = primera_resolucion.es_zona_confirmacion == 1
                fue_resuelta_quebrada = primera_resolucion.es_zona_quiebre == 1

                print(
                    "Fue resuelta el día ",
                    fecha_resolucion,
                    " cuando el mínimo fue ",
                    primera_resolucion.Low,
                    " y el máximo fue ",
                    primera_resolucion.High,
                ) if Logging else None
                if fue_resuelta_indeterminada:
                    resolucion = "indeterminado por resolverse el mismo día"
                    print(
                        "La prueba número ",
                        prueba_nro,
                        "completó la historia sin ser determinada",
                    ) if Logging else None
                elif fue_resuelta_probada:
                    resolucion = "probado"
                    print(
                        "La prueba número ",
                        prueba_nro,
                        "completó la historia probada el día ",
                        fecha_resolucion,
                    ) if Logging else None
                elif fue_resuelta_quebrada == 1:
                    resolucion = "quebrado"
                    print(
                        "La prueba número ",
                        prueba_nro,
                        "completó la historia quebrada el día ",
                        fecha_resolucion,
                    ) if Logging else None
            else:
                fecha_resolucion = np.nan
                resolucion = "indeterminado por no resolverse nunca"
                print("No llegó a resolverse en el mes.") if Logging else None
    analisis_prueba_list = []
    analisis_prueba_list.append(
        [
            ticker + str(round(valor_resistencia, 2)),
            ticker,
            valor_resistencia,
            fecha_empieza_vigencia,
            prueba_nro,
            fecha_prueba,
            fecha_resolucion,
            resolucion,
        ]
    )
    analisis_prueba = pd.DataFrame(
        analisis_prueba_list,
        columns=[
            "id_soporte",
            "ticker",
            "valor",
            "fecha_ingreso_vigencia",
            "nro_prueba_historia",
            "fecha_prueba",
            "fecha_resolucion",
            "tipo_resolucion",
        ],
    )
    analisis_prueba["fecha_prueba"] = pd.to_datetime(analisis_prueba["fecha_prueba"])
    analisis_prueba["fecha_resolucion"] = pd.to_datetime(
        analisis_prueba["fecha_resolucion"]
    )

    def save_data(df, df_all):
        df_all = pd.concat([df, df_all], axis=0)
        return df_all

    df_all = save_data(df_all, analisis_prueba)
    if resolucion == "probado":
        print(
            "La prueba número ", prueba_nro, "admite nuevas pruebas"
        ) if Logging else None
        prueba_nro = prueba_nro + 1
        dataset = dataset[dataset.Date > fecha_resolucion]
        return recorrido_resistencia(
            ticker,
            dataset,
            fecha_empieza_vigencia,
            valor_resistencia,
            rango_quebrado,
            df_all,
            prueba_nro,
            Logging,
        )
    else:
        return df_all

def recorrido_soporte(ticker, dataset, fecha_empieza_vigencia, valor_soporte, rango_quebrado, df_all, prueba_nro, Logging,):
    """
    Para un soporte, evalua su performance después (si fue probado o si fue quebrado)

    Recibe: 
        ticker: string con el código del ticker.
        dataset: dataset con la información del activo en el tiempo.
        fecha_empieza_vigencia: día en el que el soporte entra en vigor para ver si pudo ser probada.
        valor_soporte: precio de soporte.
        rango_quebrado: rango bajo el cual se dice que el soporte fue quebrado.
        df_all: dataset donde se guarda.
        prueba_nro: número de vez en la que el mismo soporte es probado
        logging: True para ir viendo en el backtesting cada una de las etapas y poder detectar errores.

    Devuelve: 
        df_all: dataset que guarda todo el registro de las pruebas de los soportes y las resistencias.
    """
    dataset = dataset[dataset.Date > fecha_empieza_vigencia].copy()
    dataset["rango_quebrado_inicia"] = valor_soporte * (1 - rango_quebrado)
    dataset["valor"] = valor_soporte
    dataset["rango_quebrado_termina"] = valor_soporte * (1 + rango_quebrado)
    dataset["es_zona_prueba"] = np.where(
        (dataset["Low"] < dataset["valor"]) & (dataset["High"] > dataset["valor"]), 1, 0
    )
    dataset["es_zona_confirmacion"] = np.where(
        (dataset["High"] > dataset["rango_quebrado_termina"]), 1, 0
    )
    dataset["es_zona_quiebre"] = np.where(
        (dataset["Low"] < dataset["rango_quebrado_inicia"]), 1, 0
    )
    mascara_probado = dataset.es_zona_prueba == 1
    mascara_quebrado = dataset.es_zona_quiebre == 1
    nunca_fue_probado_ni_quebrado = (len(dataset[mascara_probado]) == 0) & (len(dataset[mascara_quebrado]) == 0)
    nunca_fue_probado = len(dataset[mascara_probado]) == 0
    fue_quebrado = len(dataset[mascara_quebrado]) > 0
    if nunca_fue_probado_ni_quebrado:  # NO FUE PROBADO NI QUEBRADO
        resolucion = "vigente"
        fecha_prueba = np.nan
        fecha_resolucion = np.nan
        print(
            "La prueba número ", prueba_nro, "nunca se realizó en la historia"
        ) if Logging else None
    elif nunca_fue_probado:  # QUEBRADO SIN PROBARSE
        resolucion = "quebrado"
        fecha_prueba = dataset[mascara_quebrado].iloc[0].Date
        fecha_resolucion = dataset[mascara_quebrado].iloc[0].Date
        print(
            "La prueba", prueba_nro, "nunca se realizó, pero el soporte fue quebrado"
        ) if Logging else None
    else:
        fecha_prueba = dataset[mascara_probado].iloc[0].Date
        resolucion = ""
        if fue_quebrado:
            fecha_quiebre = dataset[mascara_quebrado].iloc[0].Date
            if fecha_quiebre < fecha_prueba:
                resolucion = "quebrado"
                fecha_prueba = np.nan
                fecha_resolucion = fecha_quiebre
                print(
                    "La prueba", prueba_nro, "fue quebrada antes de probarse"
                ) if Logging else None
        if resolucion != "quebrado":
            print(
                "La prueba número ",
                prueba_nro,
                "empezó el día ",
                fecha_prueba,
                " cuando el mínimo fue ",
                dataset[mascara_probado].iloc[0].Low,
                " y el máximo fue ",
                dataset[mascara_probado].iloc[0].High,
            ) if Logging else None
            mascara_resuelto = (
                (dataset.Date > fecha_prueba) & (dataset.es_zona_confirmacion == 1)
            ) | ((dataset.Date > fecha_prueba) & (dataset.es_zona_quiebre == 1))
            if len(dataset[mascara_resuelto]) > 0:  # La prueba fue resuelta
                primera_resolucion = dataset[mascara_resuelto].iloc[0]
                fecha_resolucion = primera_resolucion.Date
                fue_resuelta_indeterminada = (primera_resolucion.es_zona_confirmacion == 1) & (primera_resolucion.es_zona_quiebre == 1)
                fue_resuelta_probada = primera_resolucion.es_zona_confirmacion == 1
                fue_resuelta_quebrada = primera_resolucion.es_zona_quiebre == 1
                fecha_resolucion = fecha_resolucion
                print(
                    "Fue resuelta el día ",
                    fecha_resolucion,
                    " cuando el mínimo fue ",
                    dataset[mascara_resuelto].iloc[0].Low,
                    " y el máximo fue ",
                    dataset[mascara_resuelto].iloc[0].High,
                ) if Logging else None
                if fue_resuelta_indeterminada:
                    resolucion = "indeterminado por resolverse el mismo día"
                    print(
                        "La prueba número ",
                        prueba_nro,
                        "completó la historia sin ser determinada",
                    ) if Logging else None
                elif fue_resuelta_probada:
                    resolucion = "probado"
                    print(
                        "La prueba número ",
                        prueba_nro,
                        "completó la historia probada el día ",
                        fecha_resolucion,
                    ) if Logging else None
                elif fue_resuelta_quebrada:
                    resolucion = "quebrado"
                    print(
                        "La prueba número ",
                        prueba_nro,
                        "completó la historia quebrada el día ",
                        fecha_resolucion,
                    ) if Logging else None
            else:
                fecha_resolucion = np.nan
                resolucion = "indeterminado por no resolverse nunca"
                print("No llegó a resolverse en el mes.") if Logging else None
    analisis_prueba_list = []
    analisis_prueba_list.append(
        [
            ticker + str(round(valor_soporte, 2)),
            ticker,
            valor_soporte,
            fecha_empieza_vigencia,
            prueba_nro,
            fecha_prueba,
            fecha_resolucion,
            resolucion,
        ]
    )
    analisis_prueba = pd.DataFrame(
        analisis_prueba_list,
        columns=[
            "id_soporte",
            "ticker",
            "valor",
            "fecha_ingreso_vigencia",
            "nro_prueba_historia",
            "fecha_prueba",
            "fecha_resolucion",
            "tipo_resolucion",
        ],
    )
    analisis_prueba["fecha_prueba"] = pd.to_datetime(analisis_prueba["fecha_prueba"])
    analisis_prueba["fecha_resolucion"] = pd.to_datetime(
        analisis_prueba["fecha_resolucion"]
    )

    def save_data(df, df_all):
        df_all = pd.concat([df, df_all], axis=0)
        return df_all

    df_all = save_data(df_all, analisis_prueba)
    if resolucion == "probado":
        print(
            "La prueba número ", prueba_nro, "admite nuevas pruebas"
        ) if Logging else None
        prueba_nro = prueba_nro + 1
        dataset = dataset[dataset.Date > fecha_resolucion]
        return recorrido_soporte(
            ticker,
            dataset,
            fecha_empieza_vigencia,
            valor_soporte,
            rango_quebrado,
            df_all,
            prueba_nro,
            Logging,
        )
    else:
        return df_all
    
def seleccion_linea(dataset, fecha, precio, clase):
    """
    Selecciona para un activo en un día, el soporte o la resistencia más próxima.

    Recibe:
        dataset = base con la información del activo un año hacia atrás.
        fecha = día en el que se quiere calcular soporte o resistencia.
        precio = precio de cierre en el día del cálculo.
        clase = 's' si es soporte, 'r' si es resistencia.

    Devuelve: 
        valor de soporte/resistencia, pruebas que a la fecha ese soporte o resistencia tuvo, antigüedad en días desde que entró en vigencia.
    """
 
    mascara_iniciados_antes = dataset.fecha_ingreso_vigencia < fecha
    mascara_vigentes_hoy = dataset.tipo_resolucion == "vigente"
    mascara_quebrados_despues = (dataset.tipo_resolucion == "quebrado") & (
        dataset.fecha_resolucion > fecha
    )
    mascara_indeterminados_despues = (
        (dataset.tipo_resolucion == "indeterminado por resolverse el mismo día")
        | (dataset.tipo_resolucion == "indeterminado por no resolverse nunca")
    ) & ((dataset.fecha_resolucion > fecha) | (dataset.fecha_resolucion.isna()))
    vigentes = dataset[
        mascara_iniciados_antes
        & (
            mascara_vigentes_hoy
            | mascara_quebrados_despues
            | mascara_indeterminados_despues
        )
    ].sort_values(by=["fecha_ingreso_vigencia"])
    if clase == "s":
        try:
            vigentes = vigentes[vigentes.valor < precio].reset_index(drop=True).copy()
            valor = vigentes.iloc[vigentes["valor"].argmax()].valor
            pruebas = len(
                vigentes[(vigentes.fecha_prueba <= fecha) & (vigentes.valor == valor)]
            )
            antiguedad = (
                pd.to_datetime(fecha)
                - pd.to_datetime(vigentes.iloc[vigentes["valor"].argmax()].fecha_ingreso_vigencia)
            ).days
            return valor, pruebas, antiguedad
        except:
            return np.nan, np.nan, np.nan
    elif clase == "r":
        try:
            vigentes = vigentes[vigentes.valor > precio].reset_index(drop=True).copy()
            valor = vigentes.iloc[vigentes["valor"].argmin()].valor
            pruebas = len(
                vigentes[(vigentes.fecha_prueba <= fecha) & (vigentes.valor == valor)]
            )
            antiguedad = (
                pd.to_datetime(fecha)
                - pd.to_datetime(vigentes.iloc[vigentes["valor"].argmin()].fecha_ingreso_vigencia)
            ).days
            return valor, pruebas, antiguedad
        except:
            return np.nan, np.nan, np.nan

def calculo_sr(dataset, ticker):
    ultimo_row = dataset.tail(1)
    ventana_puntos = 3
    rango_quebrado = 0.03
    armar_soportes_con = "Close"
    armar_resistencias_con = "Close"
    Log = False
    df_all_sv, df_all_sq = calculo_soportes_historia(ticker, dataset, ventana_puntos, rango_quebrado, armar_soportes_con, Log)
    df_all_rv, df_all_rq = calculo_resistencias_historia(ticker, dataset, ventana_puntos, rango_quebrado, armar_resistencias_con, Log)
    df_all_s = pd.concat([df_all_sv, df_all_rq], axis=0)
    df_all_r = pd.concat([df_all_rv, df_all_sq], axis=0)
    resistencia = [seleccion_linea(df_all_r, f, p, "r") for f, p in zip(ultimo_row["Date"], ultimo_row["Close"])][0]
    soporte = [seleccion_linea(df_all_s, f, p, "s") for f, p in zip(ultimo_row["Date"], ultimo_row["Close"])][0]
    return resistencia, soporte

ticker = 'GGAL.BA'
base = yf.download(ticker, start=(datetime.now() - timedelta(days=365*3)), end=datetime.now(), interval='1d').dropna().reset_index()
base['Ticker'] = ticker
r, s = calculo_sr(base, ticker)

end_time = time.time()
duration = end_time - start_time
print(r, s, duration)
