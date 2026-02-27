from tools.base import BaseTool

class ConsoleTreePrinter(BaseTool):
    name = "console_tree_printer"
    description = "A tool to automate the creation of console-based tree structures in Python."
    parameters = {
        "type": "object",
        "properties": {
            "tree_type": {
                "type": "string",
                "enum": ["holiday", "binary"],
                "description": "Type of tree to print."
            },
            "height": {
                "type": "integer",
                "minimum": 1,
                "description": "Height of the tree."
            }
        },
        "required": ["tree_type", "height"]
    }

    def run(self, **kwargs) -> str:
        try:
            tree_type = kwargs.get("tree_type")
            height = kwargs.get("height")

            if tree_type == "holiday":
                return self._print_holiday_tree(height)
            elif tree_type == "binary":
                return self._print_binary_tree(height)
            else:
                return "Error: Unsupported tree type."

        except Exception as e:
            return f"Error: {str(e)}"

    def _print_holiday_tree(self, height: int) -> str:
        tree = []
        for i in range(height):
            spaces = ' ' * (height - i - 1)
            stars = '*' * (2 * i + 1)
            tree.append(f"{spaces}{stars}{spaces}")
        return "\n".join(tree)

    def _print_binary_tree(self, height: int) -> str:
        tree = []
        for i in range(height):
            level = ' ' * (height - i - 1) + ' '.join(['*' for _ in range(2 ** i)]) + ' ' * (height - i - 1)
            tree.append(level)
        return "\n".join(tree)

TOOL = ConsoleTreePrinter()