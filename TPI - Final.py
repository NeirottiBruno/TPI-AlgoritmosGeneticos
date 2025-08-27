"""
TPI: Visualización evolutiva de la red de transacciones de Bitcoin mediante algoritmos genéticos y geometría fractal.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx
import random
import math
import csv
import os
from typing import Dict, List, Tuple

# ========================= CONFIGURACIÓN =========================

EDGELIST_FILE = "elliptic_txs_edgelist.csv"

# Raíces (clusters) y extracción de subgrafos
NUM_ROOTS = 3
MAX_NODOS_REALES = 100
PROFUNDIDAD_BFS = 4  # Niveles explorados por BFS desde cada raíz

# Layout fractal
ESCALA_VALOR_DISTANCIA = 6.0
APERTURA_BASE = math.pi * 0.8       # apertura angular por nodo
FACTOR_CONTRACCION = 0.5           # cuánto se acorta la distancia por nivel

# Espaciado entre niveles de nodos
ESPACIO_POR_NIVEL = 1.5

# Operadores genéticos por porcentajes (sobre población de nodos)
PORC_MUTACION = 0.09                # ~9% de nodos candidatos sufren mutación
PORC_CROSSOVER = 0.02               # ~2% de hojas participan de crossovers
PROFUNDIDAD_MUTACION = 2            # longitud de la “cadena” mutada
RAMAS_MUTACION = 2                  # número de ramas inmediatas en mutación

# Métodos de selección para elegir nodos que mutan / cruzan: "uniforme" o "ruleta_outdeg"
SELECCION_MUTACION = "ruleta_outdeg"
SELECCION_CROSSOVER = "ruleta_outdeg"


# ========================= UTILIDADES =========================

def cargar_grafo_elliptic(archivo_csv: str) -> nx.DiGraph:
    """
    Carga un grafo dirigido (DiGraph) a partir de un CSV de aristas (origen, destino).
    Si no existe el archivo, genera un grafo aleatorio de prueba con pesos “valor”.
    Elliptic++ no trae montos reales de BTC en estos archivos; el “valor”
    se simula como un peso [0.01, 1.0] útil para layout/visual.
    """
    G = nx.DiGraph()
    if not os.path.exists(archivo_csv):
        print("Archivo CSV no encontrado. Usando grafo aleatorio de prueba.")
        G = nx.gnp_random_graph(50, 0.05, directed=True)
        for u, v in G.edges():
            G[u][v]["tipo"] = "real"
            G[u][v]["valor"] = round(random.uniform(0.01, 1.0), 8)
        return G

    with open(archivo_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # cabecera
        for row in reader:
            if len(row) < 2:
                continue
            origen, destino = row[0].strip(), row[1].strip()
            if origen != destino:
                G.add_edge(origen, destino, tipo="real", valor=round(random.uniform(0.01, 1.0), 8))
    return G


def extraer_subgrafo_jerarquico(G: nx.DiGraph, raiz: str, max_nodos: int, profundidad: int) -> nx.DiGraph:
    """
    Extrae un subgrafo “en capas” desde una raíz aplicando BFS (Breadth-First Search).
    ¿Qué es BFS? Es un recorrido por “niveles” en anchura: visita primero todos los
    vecinos a distancia 1 de la raíz, luego todos a distancia 2, y así sucesivamente.
    Se usa para:
      - Obtener un árbol/jerarquía a partir de nodos con alta actividad saliente.
      - Limitar el crecimiento mediante 'max_nodos' y 'profundidad' por rendimiento.
    """
    nodos: List[str] = [raiz]
    niveles = {raiz: 0}
    queue = [raiz]
    while queue and len(nodos) < max_nodos:
        actual = queue.pop(0)
        nivel = niveles[actual]
        if nivel >= profundidad:
            continue
        for vecino in G.successors(actual):
            if vecino not in niveles:
                niveles[vecino] = nivel + 1
                nodos.append(vecino)
                queue.append(vecino)
            if len(nodos) >= max_nodos:
                break
    return G.subgraph(nodos).copy()

def expandir_mutacion(G: nx.DiGraph, nodo: str, profundidad: int, ramas: int, prefijo: str, tipo: str = "mut", cluster: int = None) -> None:
    if profundidad == 0:
        return
    for i in range(ramas):
        nuevo = f"{prefijo}_{profundidad}_{i}_{random.randint(0,999)}"
        G.add_edge(nodo, nuevo, valor=round(random.uniform(0.01, 1.0), 8), tipo=tipo)
        if cluster is not None:
            G.nodes[nuevo]["cluster"] = cluster
        expandir_mutacion(G, nuevo, profundidad - 1, ramas, prefijo, tipo=tipo, cluster=cluster)


def seleccionar_nodos(candidatos: List[str], G: nx.DiGraph, k: int, metodo: str) -> List[str]:
    """Selecciona k candidatos (sin reemplazo) por 'uniforme' o 'ruleta_outdeg'."""
    if not candidatos:
        return []
    k = max(0, min(k, len(candidatos)))
    if metodo == "ruleta_outdeg":
        # Probabilidad proporcional a out-degree+1 (evita ceros)
        pesos = [G.out_degree(n) + 1.0 for n in candidatos]
        total = sum(pesos)
        eligidos = []
        pool = list(zip(candidatos, pesos))
        for _ in range(k):
            if not pool:
                break
            r = random.random() * total
            acum = 0.0
            for i, (n, w) in enumerate(pool):
                acum += w
                if acum >= r:
                    eligidos.append(n)
                    total -= w
                    pool.pop(i)
                    break
        return eligidos
    # uniforme
    return random.sample(candidatos, k)


def agregar_mutaciones(G, porcentaje, profundidad, ramas, metodo_sel, objetivo="out|leaves|all", solo_reales=False):
    """
    Aplica MUTACIONES sobre un % de nodos con salida (candidatos).
    La mutación agrega ramificaciones artificiales tipo 'mut'.
    """
    if objetivo == "out":
        candidatos = [n for n in G.nodes() if G.out_degree(n) > 0]
    elif objetivo == "leaves":
        candidatos = [n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0]
    else:  # "all"
        candidatos = list(G.nodes())

    if solo_reales:
        # opcional: sólo nodos que tengan al menos una arista real saliente
        candidatos = [n for n in candidatos if any(G[n][v].get("tipo") in (None,"real") for v in G.successors(n))]

    k = max(1, round(len(candidatos) * porcentaje))
    mutados = seleccionar_nodos(candidatos, G, k, metodo_sel)

    for idx, nodo in enumerate(mutados):
        cid = G.nodes[nodo].get("cluster", None)
        expandir_mutacion(G, nodo, profundidad, ramas, f"mut_{idx}", tipo="mut", cluster=cid)


def agregar_crossover(G: nx.DiGraph, pos: Dict[str, Tuple[float, float]], porcentaje: float, metodo_sel: str) -> None:
    """
    CROSSOVER entre clusters: elige pares de hojas de clusters distintos y crea un hijo
    conectado a ambos (recombinación). La cantidad de cruces se determina por porcentaje
    del total de hojas disponibles.
    """
    # Mapear nodos por cluster
    cluster_map: Dict[int, List[str]] = {}
    for n, data in G.nodes(data=True):
        cid = data.get("cluster", None)
        if cid is not None:
            cluster_map.setdefault(cid, []).append(n)

    clusters = list(cluster_map.keys())
    hojas_cluster: Dict[int, List[str]] = {cid: [n for n in cluster_map[cid] if G.out_degree(n) == 0] for cid in clusters}

    total_hojas = sum(len(v) for v in hojas_cluster.values())
    num_cross = max(1, round(total_hojas * porcentaje))

    def elegir_hoja(cid: int) -> str:
        hojas = hojas_cluster[cid]
        if not hojas:
            return None
        if metodo_sel == "ruleta_outdeg":
            # Aquí out-degree es 0; usamos in-degree+1 para variar
            pesos = [G.in_degree(n) + 1.0 for n in hojas]
            total = sum(pesos)
            r = random.random() * total
            acum = 0.0
            for i, (n, w) in enumerate(zip(hojas, pesos)):
                acum += w
                if acum >= r:
                    return n
            return hojas[-1]
        return random.choice(hojas)

    intentos = 0
    realizados = 0
    while realizados < num_cross and intentos < num_cross * 6:
        intentos += 1
        if len(clusters) < 2:
            break
        c1, c2 = random.sample(clusters, 2)
        p1, p2 = elegir_hoja(c1), elegir_hoja(c2)
        if not p1 or not p2:
            continue
        hijo = f"cross_{realizados}_{random.randint(1000,9999)}"
        G.add_node(hijo)
        G.add_edge(p1, hijo, tipo="cross", valor=round(random.uniform(0.01, 1.0), 8))
        G.add_edge(p2, hijo, tipo="cross", valor=round(random.uniform(0.01, 1.0), 8))

        # Posición del hijo entre los padres (si se conoce)
        if p1 in pos and p2 in pos:
            x1, y1 = pos[p1]
            x2, y2 = pos[p2]
            pos[hijo] = ((x1 + x2) / 2 + random.uniform(-0.5, 0.5),
                         (y1 + y2) / 2 + random.uniform(-0.5, 0.5))
        realizados += 1


# ========================= LAYOUT FRACTAL =========================

def generar_layout_fractal_multiple(G: nx.DiGraph, roots: List[str], radio_inicial: float = 5.0, max_nivel: int = 6) -> Dict[str, Tuple[float, float]]:
    """
    Genera posiciones XY “tipo árbol fractal” por raíz (cluster) alrededor de un círculo central.
    """
    pos: Dict[str, Tuple[float, float]] = {}
    R = radio_inicial * 3.0
    step = 2 * math.pi / max(1, len(roots))
    for i, root in enumerate(roots):
        ang = i * step
        pos[root] = (math.cos(ang) * R, math.sin(ang) * R)
        posicionar_fractal(G, root, pos, ang, radio_inicial, 1, max_nivel)
    return pos


def posicionar_fractal(G: nx.DiGraph, nodo: str, pos: Dict[str, Tuple[float, float]], angulo: float, radio: float, nivel: int, max_nivel: int) -> None:
    """
    Posiciona hijos recursivamente en abanico con:
    - apertura angular decreciente con la profundidad (autosimilitud visual)
    - distancia radial proporcional al peso de la arista y a FACTOR_CONTRACCION
    """
    if nivel > max_nivel:
        return
    hijos = list(G.successors(nodo))
    if not hijos:
        return

    hijos = sorted(hijos, key=str)  # orden estable para reproducibilidad visual
    apertura_total = APERTURA_BASE * (FACTOR_CONTRACCION ** (nivel - 1))
    paso = apertura_total / max(1, len(hijos) - 1) if len(hijos) > 1 else 0
    ang_inicial = angulo - apertura_total / 2.0

    for i, h in enumerate(hijos):
        a = ang_inicial + i * paso
        valor = G[nodo][h].get("valor", 0.5)
        distancia = radio * (0.5 + valor) * ESCALA_VALOR_DISTANCIA + nivel * ESPACIO_POR_NIVEL
        x = pos[nodo][0] + math.cos(a) * distancia + (random.uniform(-1, 1))
        y = pos[nodo][1] + math.sin(a) * distancia + (random.uniform(-1, 1))
        pos[h] = (x, y)
        posicionar_fractal(G, h, pos, a, radio * FACTOR_CONTRACCION, nivel + 1, max_nivel)


# ========================= PIPELINE PRINCIPAL =========================

def main():
    """
    Pipeline principal:
      1) Carga/crea grafo de transacciones (población = todos los nodos).
      2) Selecciona raíces por mayor out-degree (clusters).
      3) Extrae subgrafos vía BFS (jerarquía). Cada nodo es un individuo.
      4) Refuerza autosimilitud del árbol en cada cluster (estructura fractal clara).
      5) Aplica mutaciones (% de nodos) y crossovers (% de hojas entre clusters).
      6) Calcula layout fractal y dibuja.
    """
    G_total = cargar_grafo_elliptic(EDGELIST_FILE)

    # Raíces: mayores emisores (out-degree)
    roots = [n for n, d in sorted(G_total.out_degree(), key=lambda x: x[1], reverse=True)[:NUM_ROOTS]]

    # Construir subgrafo compuesto por clusters
    subG = nx.DiGraph()
    for cid, root in enumerate(roots):
        sub = extraer_subgrafo_jerarquico(G_total, root, max_nodos=MAX_NODOS_REALES, profundidad=PROFUNDIDAD_BFS)
        for n in sub.nodes():
            sub.nodes[n]["cluster"] = cid
        subG = nx.compose(subG, sub)


    # Operadores genéticos con porcentajes
    agregar_mutaciones(subG, PORC_MUTACION, PROFUNDIDAD_MUTACION, RAMAS_MUTACION, metodo_sel=SELECCION_MUTACION, objetivo="leaves") 

    # Layout antes del crossover (para posicionar los hijos “entre” padres)
    pos = generar_layout_fractal_multiple(subG, roots, radio_inicial=3.0)

    # Crossover
    agregar_crossover(subG, pos, PORC_CROSSOVER, metodo_sel=SELECCION_CROSSOVER)

    # --------------------------------------- Visualización ---------------------------------------
    colores = ["skyblue", "lightgreen", "lightpink", "lightsalmon"]
    node_colors = [colores[subG.nodes[n].get("cluster", -1) % len(colores)] for n in subG.nodes()]

    edges_real = [e for e in subG.edges() if subG.edges[e].get("tipo") == "real"]
    edges_mut = [e for e in subG.edges() if subG.edges[e].get("tipo") == "mut"]
    edges_cross = [e for e in subG.edges() if subG.edges[e].get("tipo") == "cross"]

    plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(subG, pos, node_color=node_colors, node_size=300)
    nx.draw_networkx_edges(subG, pos, edgelist=edges_real, edge_color="gray", arrowsize=6, alpha=0.5)
    nx.draw_networkx_edges(subG, pos, edgelist=edges_mut, edge_color="orange", arrowsize=6, alpha=0.7)
    nx.draw_networkx_edges(subG, pos, edgelist=edges_cross, edge_color="red", arrowsize=6, alpha=0.85)

    patches = [
        mpatches.Patch(color="skyblue", label="Cluster 1"),
        mpatches.Patch(color="lightgreen", label="Cluster 2"),
        mpatches.Patch(color="lightpink", label="Cluster 3"),
    ]
    lines = [
        mlines.Line2D([], [], color="gray", label="Transacción", linewidth=2),
        mlines.Line2D([], [], color="orange", label="Mutación", linewidth=2),
        mlines.Line2D([], [], color="red", label="Crossover", linewidth=2),
    ]
    plt.legend(handles=patches + lines, loc="upper right", fontsize=10, frameon=True)
    plt.title("Red fractal evolutiva de Bitcoin con operadores genéticos, en base a datos Elliptic++", fontsize=15)
    plt.axis("off")
    plt.tight_layout()

    plt.savefig("red_fractal_evolutiva.png", dpi=220, bbox_inches="tight")
    plt.close()
    print("Imagen guardada en red_fractal_evolutiva.png")


if __name__ == "__main__":
    main()