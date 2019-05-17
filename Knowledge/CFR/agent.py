"""

树模型： 不是二叉树
目的: 制作一个通用的树，了解树模型的构造和为什么占用巨大资源

有两类node， state 和 decision node
"""

import numpy as np

class Node:

    """
    node
    left right Node类
    """

    def __init__(self, is_root=False):
        self.is_root = is_root
        self.children = []

    def add_children(self, node):
        self.children.append(node)

    @property
    def isend(self):
        """判断是否是一棵树的end"""
        # bool([]) is False
        return bool(self.children)

class StateNode(Node):

    def __init__(self, is_root=False):
        super(StateNode, self).__init__(is_root)
        self.type = "state"

    def add_children(self, node):
        """state 下来必须是decision？"""
        assert node.type is "dec"
        self.children.append(node)

    @property
    def isstate(self):
        return self.type is "state"


class DecNode(Node):

    def __init__(self, is_root=False):
        super(DecNode, self).__init__(is_root)
        self.type = "dev"

    def add_children(self, node):
        """decision下来必须是state ？"""
        assert node.type is "state"
        self.children.append(node)

    @property
    def isstate(self):
        return self.type is "dec"


class Tree:
    def __init__(self, debug=False):
        self.root = Node(True)
        self.debug = debug

    def lookup(self, h):
        assert not self.root.isend, "root未分裂"
        next_node = self.root

        while not next_node.isend:
            node = next_node

            next_end = node.isend
        return node.y_value.mean()

    def generate(self, node, data):
        # 数据不是空的
        for child in node.children:
            # 分裂节点
            data = data
            if not data:
                self.generate(child)
            else:
                return
