import re, bz2
from xml.etree import ElementTree

NAMESPACE_PREFIX_PTN = re.compile(r"^{(?P<namespace>[^}]+)}(?P<node_class>.+)$")
def node_class(full_tag):
    m = NAMESPACE_PREFIX_PTN.fullmatch(full_tag)
    return m.group("node_class") if m is not None else full_tag

def node_childs(node):
    normalized_childs = {}
    for child in node:
        key = node_class(child.tag)
        if key not in normalized_childs:
            normalized_childs[key] = child
    return normalized_childs

class Page:
    def __init__(self, title: str, redirect: str, text: str):
        self.title = title
        self.redirect = redirect
        self.text = text

    @classmethod
    def from_xml(cls, xml_node):
        if node_class(xml_node.tag) != "page":
            return None

        ncs = node_childs(xml_node)

        try: ns = ncs.get("ns").text
        except: ns = None

        # ignore non-default namespaces
        if ns != "0":
            return None

        try: title = ncs.get("title").text
        except: title = None

        if not title: # or not has_valid_prefix(title):
            return None

        # extract text
        text = None
        revision_node = ncs.get("revision")
        if revision_node is not None:
            revision_childs = node_childs(revision_node)
            text_node = revision_childs.get("text")
            if text_node is not None:
                text = text_node.text

        # extract redirect
        redirect = None
        redirect_node = ncs.get("redirect")
        if redirect_node is not None:
            redirect = redirect_node.attrib.get("title", None)

        if redirect is None and text is None:
            return None

        return Page(title, redirect=redirect, text=text)

    def __repr__(self):
        fmt_redirect = f" -> {repr(self.redirect)}" if self.redirect else ""
        return f"<Page {repr(self.title)}{fmt_redirect}>"

    @classmethod
    def load_from_dumpfile(cls, dumpfile: str, offset: int = 0, skip_after: int = 1):
        # open file
        with bz2.open(dumpfile, "rt") as page_file:
            # peephole parsing
            parser = ElementTree.iterparse(page_file, events=("start", "end"))
            n_skipping = offset

            # do parsing
            inNode = False
            for event, node in parser:
                if event == "start" and node_class(node.tag) == "page":
                    # entered page section
                    inNode=True

                if event == "end" and node_class(node.tag) == "page":
                    # reached end of page section
                    # hence: process data and clear nodes
                    if n_skipping == 0:
                        yield Page.from_xml(node)
                        n_skipping = skip_after

                    n_skipping -= 1
                    inNode=False

                if not inNode:
                    # cleanup parsing window
                    node.clear()
