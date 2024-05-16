import asyncio

# Example async functions
async def multiply_function(node, c):
    print("Multiplying {} by -1".format(c))
    await asyncio.sleep(1)
    b = c * -1
    print("Result is b =", b)
    return {"b": b}

# Example async functions
async def add_function(node, a, b):
    print(f"Adding {a} and {b}")
    await asyncio.sleep(1)
    c = a + b
    print("Result is c =", c)
    return {"c": c}
    
# Dictionary containing functions
functions = {
    "multiply_by_negative_one": multiply_function,
    "add": add_function,
    # Add more functions here as needed
}

class AsyncNode:
    def __init__(self, function_name, input_addresses=None, output_args=None):
        self.trigger_in = None
        self.trigger_out = []
        self.function_name = function_name
        self.input_addresses = input_addresses or []
        self.output_args = output_args or {}

    async def trigger(self):
        if self.trigger_in is not None:
            print("Triggering input node")
            await self.trigger_in.trigger()

        # Get the function from the dictionary based on the function_name
        function_to_call = functions.get(self.function_name)
        if function_to_call:
            print(f"Calling function {self.function_name}")
            # Fetch input_args from input_addresses
            input_args = {}
            for address in self.input_addresses:
                node = address.get("node")
                arg_name = address.get("arg_name")
                input_args[arg_name] = node.output_args.get(arg_name)

            # Pass input_args and self to the function
            output_args = await function_to_call(self, **input_args)
            print(output_args)

            # Update output_args with the function's output, appending new args and replacing existing ones
            for arg_name, value in output_args.items():
                if arg_name not in self.output_args:
                    self.output_args[arg_name] = value
                else:
                    self.output_args[arg_name] = value
        #print(node)
        print(self.output_args)
        for node in self.trigger_out:
            print(f"Triggering output node {node.function_name}")
            await node.trigger()

# Example usage
async def main():
    source = AsyncNode(None, input_addresses=[], output_args={"a": 5, "b": 3})
    node1 = AsyncNode("add", input_addresses=[{"node": source, "arg_name": "a"}, {"node": source, "arg_name": "b"}], output_args={"a": 5})
    node2 = AsyncNode("multiply_by_negative_one", input_addresses=[{"node": node1, "arg_name": "c"}], output_args={"a": 0, "b": 0})
    node1.trigger_out = [node2]
    node2.trigger_out = []  # This line might be redundant since trigger_out is already initialized as an empty list in __init__
    node1.trigger_in = None  # Assuming this is the initial trigger
    #node2.trigger_in = node1
    await node1.trigger()
    
    source.output_args["a"] = 6
    
    print(source.output_args)
    print("Output args of node1:", node1.output_args)
    print("Output args of node2:", node2.output_args)

asyncio.run(main())
