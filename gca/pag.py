from causallearn.utils.GraphUtils import GraphUtils
from IPython.display import display, SVG

from gca.data.descriptors import DESCRIPTORS
from gca.data.tasks import COLORS
from gca.utils import create_prettytable, get_graph_for_print, subtitle, title


def to_svg(graph, x_map, dtypes, cols, descriptors, color=False, color_map=COLORS):
    pdot = GraphUtils.to_pydot(graph, labels=cols)

    for node in pdot.get_nodes():
        label = node.get_label()
        label = x_map[label] if label in x_map.keys() else label
        if label in dtypes:
            node.set_tooltip(f'{descriptors[label]} | dtype={str(dtypes[label])}')
        if label.startswith('PRED'):
            node.set_style('filled')  # bold?
            node.set_fillcolor('#ececec')
        elif color and label in color_map:
            node.set_style('filled')
            node.set_fillcolor(color_map[label])

    def node_name_by_id(id):
        if isinstance(id, int):
            id = str(id)
        label = pdot.obj_dict['nodes'][id][0]['attributes']['label']
        return x_map[label] if label in x_map else label

    def explain_tooltip(src, dest, tail, head):
        tips = {tail, head}
        if tips == {'normal'}:
            return f'<-> {src} and {dest} share a latent confounder U (not in graph)'
        elif tips == {'none', 'normal'}:
            if head == 'none':
                dest, src = src, dest
            return f'--> {src} causes {dest}'
        elif tips == {'odot', 'normal'}:
            return f'1. {explain_tooltip(src, dest, "none", "normal")} OR 2. {explain_tooltip(src, dest, "normal", "normal")}'
        elif tips == {'odot'}:
            return f'No set m-separates {src} and {dest}; 1. {explain_tooltip(src, dest, "none", "normal")} OR 2. {explain_tooltip(src, dest, "normal", "none")} OR 3. {explain_tooltip(src, dest, "normal", "normal")}'
        return 'UNK'


    for edge in pdot.get_edges():
        src, dest = node_name_by_id(edge.get_source()), node_name_by_id(edge.get_destination())
        tail, head = edge.get_arrowtail(), edge.get_arrowhead()
        tooltip = explain_tooltip(src, dest, tail, head)
        edge.set_edgetooltip(tooltip)
        edge.set_headtooltip(tooltip)
        edge.set_tailtooltip(tooltip)

    return pdot.create_svg()


def create_pag(df,
               alpha=0.05,
               depth=-1,
               verbose=True,
               verbosity_level=1,
               show_progress=True,
               color=False,
               view_svg=False,
               descriptors=DESCRIPTORS):
    import io
    import re
    from contextlib import redirect_stdout

    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import chisq
    from causallearn.graph.GraphNode import GraphNode
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    # Y_hat (Xn) cannot be a causal ancestor of X[0, ..., n-1]
    dtypes, cols = df.dtypes, df.columns.to_list()
    x_map = {f'X{i}': col for i, col in enumerate(cols, start=1)}
    x_pred = f'X{len(df.columns)}'
    background_knowledge = BackgroundKnowledge().add_forbidden_by_pattern(x_pred, f'X[1-{len(df.columns) - 1}]')

    # FCI with Chi-Squared (default alpha=0.05)
    with redirect_stdout(io.StringIO()) as buf:
        G, edges = fci(dataset=df.to_numpy(), 
                       depth=depth,
                       independence_test_method=chisq,
                       alpha=alpha,
                       background_knowledge=background_knowledge,
                       verbose=True,
                       show_progress=show_progress)

    pattern = re.compile('^(\d+) (dep|ind) (\d+) \| \((.*?)\) with p-value (.*)')

    def extract_results(line):
          res = pattern.search(line)
          return {'X': x_map[f'X{int(res.group(1)) + 1}'],
                  'Y': x_map[f'X{int(res.group(3)) + 1}'],
                  'S': [] if not res.group(4) else [x_map[f'X{int(x) + 1}'] for x in res.group(4).strip(', ').split(', ')],
                  'p-value': float(res.group(5)),
                  'independent': res.group(2) == 'ind'}

    test_results = [extract_results(line) for line in buf.getvalue().splitlines() if 'with p-value' in line]
    sanity_check = G.get_outdegree(GraphNode(x_pred)) == 0

    if verbose and verbosity_level > 1:
        print(title(f'FCI WITH CHI-SQUARED (alpha={alpha})'))
        print('Results of fast adjacency search (FAS)\n\t... where X _||_ Y | S\n')
        table = create_prettytable(test_results)
        table.align['p-value'] = 'l'
        print(table)

        print(subtitle('X_MAP'))
        print(x_map)

    def extract_allowed_forbidden(bgk):
        def nodes_to_arrows(specs):
            return '; '.join(f'{x} *-> {y}' for x, y in specs) if specs else None

        def pattern_to_arrows(specs):
            return '; '.join([f'pattern "{x} *-> {y}"' for x, y in specs]) if specs else None

        required = ' and '.join(filter(None, [nodes_to_arrows(bgk.required_rules_specs), pattern_to_arrows(bgk.required_pattern_rules_specs)]))
        forbidden = ' and '.join(filter(None, [nodes_to_arrows(bgk.forbidden_rules_specs), pattern_to_arrows(bgk.forbidden_pattern_rules_specs)]))
        return f'Required: {required if required else "-"}\nForbidden: {forbidden if forbidden else "-"}'

    if verbose:
        title_to_apply = subtitle if verbosity_level > 1 else title
        print(title_to_apply('BACKGROUND KNOWLEGE'))
        print(extract_allowed_forbidden(background_knowledge))

    if verbose:
        print(title('FINAL GRAPH'))
        print('---{graph.txt}---')
        print(get_graph_for_print(G, x_map))
        print('-' * 17)

    svg = to_svg(G, x_map, dtypes, cols, descriptors, color=color)

    if verbose:
        print(subtitle('SANITY CHECKS'))
        print(f'  1. Correct x_map ({x_pred} is last and startswith "PRED_"):', x_map[x_pred].startswith("PRED_"))
        print(f'  2. {x_pred} [{x_map[x_pred]}] has no outgoing arrows (outdegree == 0):', sanity_check)

        if view_svg:
            print(title('GRAPH IMAGE'))

    if view_svg:
        display(SVG(svg))

    return G, edges, x_map, test_results, sanity_check, svg
