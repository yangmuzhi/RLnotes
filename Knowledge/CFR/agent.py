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
    def isdec(self):
        return self.type is "dec"

# game tree

class Tree:
    """
    root 是一个node类，游戏初始化后，child 添加 state
    """
    def __init__(self, debug=False):
        self.root = Node(True)
        self.debug = debug

    # def lookup(self, h):
    #     assert not self.root.isend, "root未分裂"
    #     next_node = self.root
    #
    #     while not next_node.isend:
    #         node = next_node
    #
    #         next_end = node.isend
    #     return node.y_value.mean()

    def look_node_exist(self, h_dict):

        return

    def add_decnode(self, node):
        assert node.type == "dec"
        return

    def add_statenode(self, node):
        assert node.type == "state"
        return

    def generate(self, node, h_dict):
        """根据h_dict一个一个添加进去 """

        # 数据不是空的

        if self.look_node_exist():
            pass
        for child in node.children:
            # 分裂节点
            data = data
            if not data:
                self.generate(child)
            else:
                return
