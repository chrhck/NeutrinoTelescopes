import torch
import jammy_flows
import numpy as np


def iter_graph(root, callback):
    queue = [(None, root)]
    seen = set()
    while queue:
        prev_fn, fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append((fn, next_fn))
        callback(prev_fn, fn)


def register_hooks(var):
    fn_dict = {}

    def hook_cb(prev_fn, fn):
        def register_grad(grad_input, grad_output):
            # print(prev_fn, fn, fn.next_functions)
            fn_dict[fn] = False
            if any(t is not None and torch.any(~torch.isfinite(t)) for t in grad_input):
                # print(f"{fn} grad_input={grad_input} grad_output={grad_output}")
                fn_dict[fn] = True
            if any(
                t is not None and torch.any(~torch.isfinite(t)) for t in grad_output
            ):
                fn_dict[fn] = True
                # print(f"{fn} grad_input={grad_input} grad_output={grad_output}")

            # if all(t is not None and torch.all(torch.isfinite(t)) for t in grad_input) and any(t is not None and torch.any(~torch.isfinite(t)) for t in grad_output):
            #    print(f"{fn} grad_input={grad_input} grad_output={grad_output}")

            # assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_input), f"{fn} grad_input={grad_input} grad_output={grad_output}"
            # assert all(t is None or torch.all(~torch.isnan(t)) for t in grad_output), f"{fn} grad_input={grad_input} grad_output={grad_output}"

        fn.register_hook(register_grad)

    iter_graph(var.grad_fn, hook_cb)
    return fn_dict


def find_nan_root(var, fn_dict):
    def nan_cb(prev_fn, fn):

        if fn_dict.get(fn, False) and not fn_dict.get(prev_fn, False):
            print(f"Candidate: {fn} {prev_fn}")

        if not fn_dict.get(fn, False) and fn_dict.get(prev_fn, False):
            print(f"Candidate: {fn}")

    iter_graph(var.grad_fn, nan_cb)







data = torch.load(
    "/home/chrhck/repos/NeutrinoTelescopes/assets/models/checkpoint_3_nangrad.pt"
)
model_state_dict = data["model_state_dict"]
optimizer_state_dict = data["optimizer_state_dict"]
samples, labels = data["batch"]
extra_flow_defs = {
    "v": {
        "exp_map_type": "splines",
        #"nonlinear_stretch_type": "rq_splines"
    }
}
pdf = jammy_flows.pdf("e1+s2", "gg+n", conditional_input_dim=5, hidden_mlp_dims_sub_pdfs="256", options_overwrite=extra_flow_defs)
pdf.load_state_dict(model_state_dict)
pdf.to("cpu")

# 500
slrange = slice(0, 1000)

samples = samples[slrange, :].to("cpu")
labels = labels[slrange, :].to("cpu")



samples[1, :] = torch.ones_like(samples[1, :])
labels[1, :] = torch.ones_like(labels[1, :])
inp = samples[:, :3]
w = samples[:, 3] * samples.shape[0] / sum(samples[:, 3])

with torch.autograd.detect_anomaly():
    transformed_input = pdf.transform_target_into_returnable_params(inp)
    log_pdf, _, _ = pdf(
        transformed_input, conditional_input=labels, force_embedding_coordinates=True
    )
    neg_log_loss = (-log_pdf * w).mean()

    neg_log_loss.backward()

    for p in pdf.parameters():
        if torch.any(~torch.isfinite(p.grad)):
            print(f"Error in {p}")


# find_nan_root(neg_log_loss, fn_dict)
# fn_dict = register_hooks(neg_log_loss)
# draw_graph(neg_log_loss)


"""
for name, p in pdf.named_parameters(recurse=True):
    if p.grad is not None:
        if torch.any(~torch.isfinite(p.grad)):
            print(name, np.argmin(np.isfinite(p.grad.detach().cpu().numpy()), axis=0))
            print(p.grad, np.argmin(np.isfinite(p.grad.detach().cpu().numpy()), axis=0))
"""


def draw_graph(start, watch=None):
    if watch is None:
        watch = []
    from graphviz import Digraph

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
    )
    graph = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    assert hasattr(start, "grad_fn")
    if start.grad_fn is not None:
        _draw_graph(start.grad_fn, graph, watch=watch)

    size_per_element = 0.15
    min_size = 12

    # Get the approximate number of nodes and edges
    num_rows = len(graph.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    graph.graph_attr.update(size=size_str)
    graph.render(filename="net_graph.jpg")


def _draw_graph(var, graph, watch=None, seen=None, indent="", pobj=None):
    if watch is None:
        watch = []
    if seen is None:
        seen = []
    """ recursive function going through the hierarchical graph printing off
    what we need to see what autograd is doing."""
    from rich import print

    if hasattr(var, "next_functions"):
        for fun in var.next_functions:
            joy = fun[0]
            if joy is None:
                continue
            if joy in seen:
                continue

            label = (
                str(type(joy)).replace("class", "").replace("'", "").replace(" ", "")
            )
            label_graph = label
            colour_graph = ""
            seen.append(joy)

            if hasattr(joy, "variable"):
                happy = joy.variable
                if happy.is_leaf:
                    label += " \U0001F343"
                    colour_graph = "green"

                    for (name, obj) in watch:
                        if obj is happy:
                            label += (
                                " \U000023E9 "
                                + "[b][u][color=#FF00FF]"
                                + name
                                + "[/color][/u][/b]"
                            )
                            label_graph += name

                            colour_graph = "blue"
                            break

                        vv = [str(obj.shape[x]) for x in range(len(obj.shape))]
                        label += " [["
                        label += ", ".join(vv)
                        label += "]]"
                        label += " " + str(happy.var())

            graph.node(str(joy), label_graph, fillcolor=colour_graph)
            print(indent + label)
            _draw_graph(joy, graph, watch, seen, indent + ".", joy)
            if pobj is not None:
                graph.edge(str(pobj), str(joy))
