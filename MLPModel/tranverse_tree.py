import os
import javalang
from javalang.ast import Node
import numpy as np


def _name(node):
    return type(node).__name__


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'
    elif isinstance(node, Node):
        token = node.__class__.__name__
    return token


def dfsSearch1(children):
    if not isinstance(children, (str, Node, list, tuple)):
        return
    if isinstance(children, (str, Node)):
        if str(children) == '':
            return
        # ss = str(children)
        if str(children).startswith('"'):
            return
        if str(children).startswith("'"):
            return
        if str(children).startswith("/*"):
            return
        global num_nodes
        num_nodes += 1
        listt1.append(children)
        return
    for child in children:
        if isinstance(child, (str, Node, list, tuple)):
            dfsSearch1(child)


def _traverse_tree(root):
    global num_nodes
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)

        global listt
        global listt1
        listt1 = []
        dfsSearch1(current_node.children)
        children = listt1
        for child in children:
            child_json = {
                "node": get_token(child),
                "children": []
            }

            current_node_json['children'].append(child_json)
            if isinstance(child, (Node)):
                queue_json.append(child_json)
                queue.append(child)
    return root_json, num_nodes


def dfsDict(root, listtfinal):
    listtfinal.append(str(root['node']))
    if len(root['children']):
        pass
    else:
        return
    for dictt in root['children']:
        dfsDict(dictt, listtfinal)