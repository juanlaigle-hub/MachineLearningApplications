# Machine Learning Applications · ML industrial con física

**Sitio en vivo:** https://juanlaigle-hub.github.io/MachineLearningApplications/ · [English version](README.md)

Dos problemas reales de planta resueltos de punta a punta: datos generados con simuladores físicos validados,
chequeos de fenomenología, entrenamiento y validación honesta, exportación del modelo y una demo en vivo que corre en
el navegador.

| | Proyecto | Problema industrial | Resultado | Demo en vivo |
|---|---|---|---|---|
| ![](assets/clasificador.gif) | **[Diagnóstico de cartas dinamométricas](ml_cartas_dinamometricas/)** | Oil & gas: clasificar cada carta de fondo de un pozo con bombeo mecánico en 7 condiciones (golpe de fluido, gas, fugas en válvulas, varilla cortada, golpe abajo) | **98,7 %** de exactitud en prueba, **96,7 %** en pozos fuera de rango (reglas de ingeniero: 50 %) | [Abrir](https://juanlaigle-hub.github.io/MachineLearningApplications/ml_cartas_dinamometricas/clasificador.html) |
| ![](assets/sensor_virtual.gif) | **[Sensor virtual de jugo concentrado](ml_sensor_virtual_evaporador/)** | Alimentos: estimar cada minuto el °Brix (respaldo del refractómetro), el HMF y el color en un evaporador de 5 efectos de jugo de manzana | Error de °Brix **1,34** en un ensayo no visto (ingenuo: 6,45); HMF con **R² 0,93** a partir de 120 análisis | [Abrir](https://juanlaigle-hub.github.io/MachineLearningApplications/ml_sensor_virtual_evaporador/sensor_virtual.html) |

## Decisiones de ingeniería

- **La física primero.** Los rasgos salen de la física: carga de fluido y relación de carreras en la bomba; elevación
  del punto de ebullición, balance de masa y Arrhenius × tiempo de residencia en el evaporador. El modelo de °Brix
  tiene restricciones de signo, así que no puede aprender que más vapor baja el °Brix.
- **Carta de fondo, no de superficie.** El mismo modelo entrenado con la carta de superficie cae al **34 %** en pozos no
  vistos, porque esa carta depende de la dinámica de la sarta de cada pozo. Los datos confirman por qué la industria
  diagnostica con la carta de fondo, calculada con la ecuación de onda.
- **Caja gris antes que caja negra cuando faltan etiquetas.** Con solo 20 análisis de laboratorio, el modelo de caja
  gris de HMF ya rinde lo máximo (MAE 0,43 mg/kg), mientras que un gradient boosting necesita 120 para llegar a 0,46.
  Además recupera la constante cinética (0,162 frente a 0,18 real) y extrapola bien con la temperatura.
- **Elegir el modelo sin mirar la prueba.** Validación cruzada agrupada por pozo o campaña, solo entre los modelos que
  pasan las pruebas físicas. Un Ridge con error parecido quedó descartado porque su coeficiente de balance de masa
  tenía el signo equivocado.
- **Ensayos en planta.** En lazo cerrado el °Brix casi no varía y los datos no enseñan causa y efecto. Se resolvió con
  escalones de alimentación en manual, planificados para no activar los enclavamientos.
- **Los chequeos encuentran errores.** Las pruebas de fenomenología detectaron tres errores en los propios
  simuladores, y se corrigieron antes de entrenar:
  - una fuga que *aumentaba* la producción;
  - un °Brix que bajaba a lo largo del evaporador;
  - una cadena de energía con dos soluciones.

## Del dato al despliegue

Cada README de proyecto tiene una tabla de 12 pasos que compara cómo se hace en una planta real con lo que se hizo
aquí:
1. Problema y KPI.
2. Extracción del historiador.
3. Etiquetado o datos de laboratorio.
4. Ensayos en planta.
5. Limpieza y alineación.
6. Rasgos con física.
7. Partición por pozo o por tiempo.
8. Líneas base y comparación de modelos.
9. Validación física.
10. Exportación con verificación exacta.
11. Umbrales de confianza y respaldos de operación.
12. Monitoreo de la deriva.

## Estructura

```
ml_cartas_dinamometricas/       generador del dataset, notebook ejecutado, modelo exportado, página en vivo, video
ml_sensor_virtual_evaporador/   generador del historiador y del laboratorio, notebook ejecutado, modelo, página, video
assets/                         previews
index.html                      portada del portafolio (GitHub Pages)
```

Los simuladores físicos 3D que generan los datos están en **[Simulations3D](https://github.com/juanlaigle-hub/Simulations3D)**.

## Notas

- Los modelos se entrenaron con datos **simulados**. Antes de usarlos en operación hay que validarlos y reentrenarlos
  con datos reales de campo o de planta. Cada README detalla sus limitaciones.
- Licencia MIT.
