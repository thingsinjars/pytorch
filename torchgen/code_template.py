import re
from typing import Mapping, Match, Optional, Sequence

# match $identifier or ${identifier} and replace with value in env
# If this identifier is at the beginning of whitespace on a line
# and its value is a list then it is treated as
# block substitution by indenting to that depth and putting each element
# of the list on its own line
# if the identifier is on a line starting with non-whitespace and a list
# then it is comma separated ${,foo} will insert a comma before the list
# if this list is not empty and ${foo,} will insert one after.


class CodeTemplate:
    substitution_str = r"(^[^\n\S]*)?\$([^\d\W]\w*|\{,?[^\d\W]\w*\,?})"
    substitution = re.compile(substitution_str, re.MULTILINE)

    pattern: str
    filename: str

    @staticmethod
    def from_file(filename: str) -> "CodeTemplate":
        """
        This function takes a filename as an argument and reads the contents of
        that file. Then it returns a "CodeTemplate" object created from the read
        content of the file.

        Args:
            filename (str): The `filename` parameter is the path to a file that
                will be read and passed to the `CodeTemplate` constructor as its
                `source` argument.

        Returns:
            "CodeTemplate": The output of this function is a CodeTemplate object.

        """
        with open(filename) as f:
            return CodeTemplate(f.read(), filename)

    def __init__(self, pattern: str, filename: str = "") -> None:
        """
        This function initiates an object with a given pattern and filename. It
        sets the objects 'pattern' attribute to the given pattern string and
        optionally the 'filename' attribute to the given file name.

        Args:
            pattern (str): The `pattern` input parameter is a required string
                parameter that represents a regular expression or a string to
                search for when searching for text files within the specified
                directory or directories given by `filename`.
            filename (""): The `filename` input parameter is an optional string
                that specifies the name of a file containing the pattern for which
                matches are to be found. If not provided , the program will read
                from standard input by default .

        """
        self.pattern = pattern
        self.filename = filename

    def substitute(
        self, env: Optional[Mapping[str, object]] = None, **kwargs: object
    ) -> str:
        """
        This function takes a pattern and a dictionary of substitutions as arguments
        and returns the replaced pattern with placeholders substituted using the
        values from the dictionary. It supports nested placeholders (e.g., "{{
        Foo.bar }}" syntax) and recursively replaces values if they are lists or
        dictionaries.

        Args:
            env (None): The input parameter `env` takes an optional dictionary of
                values to substitute key-value pairs with.
            	-*kwargs (object): **kwargs is used as a dictionary-like object that
                contains keyword arguments for the substitute() function. It allows
                the function to accept named arguments when called with a dictionary
                of key-value pairs (e.g., replace(match=…)).  This functionality
                enables more flexible and modular code writing since you don’t
                need to predefine every argument.

        Returns:
            str: The output of the given function is a string obtained after
            replacing all the matches of the pattern `{$ key $}` with the corresponding
            value of the specified key from an environment map or keyword arguments.
            The replacement is done using the `replace()` function which checks
            for the existence of `indent` and returns an indented list if necessary.

        """
        if env is None:
            env = {}

        def lookup(v: str) -> object:
            """
            This function takes a string `v` and looks up its value either inside
            the keyword argument `kwargs` if it's present as a key or inside the
            dictionary `env`. It returns the value found.

            Args:
                v (str): Here's an explanation of what v does:
                    
                    V is the input parameter for the function "lookup" and represents
                    a string value that is passed to the function. The function
                    uses this value to search for a matching key either inside a
                    dictionary "kwargs" or inside another dictionary "env", returning
                    the value associated with that key if found or raising an
                    assertiom failing if not.

            Returns:
                object: The function "lookup" takes a string parameter "v" and
                returns the object associated with that key. It first checks if
                the key is present int he dictionary "kwargs", if it is then it
                returns the value associated with that key. If the key is not
                present int he "kwargs", it looks for the key  "v"in the dictionary
                "env".If it found tha key ,it returnsthe associated object else None.

            """
            assert env is not None
            return kwargs[v] if v in kwargs else env[v]

        def indent_lines(indent: str, v: Sequence[object]) -> str:
            """
            The given function takes an indentation string `indent` and any sequence
            of objects `v`, and returns a string by joining each element of `v`
            with the specified indentation and adding a newline character between
            them. It removes any extra newlines from the end of the resulting string.

            Args:
                indent (str): The `indent` parameter is used to prepend a common
                    indent string to each line of the returned list.
                v (Sequence[object]): Here's the documentation string for indent_lines:
                    
                    ```
                    def indent_lines(indent: str,..., v: Sequence[object]]) -> str:
                                ...:
                    ```
                    
                    According to the parameter list`, `v` is an input parameter
                    and it's type hinted as Sequence[object]. Therefore`, `v`  is
                    a sequence of objects.

            Returns:
                str: The function takes two arguments `indent` and `v`. The return
                value is a string concatenated from each line of every sequence
                object `e`. Spaces are added to leading edges based on the argument
                `indent`. No blank lines are included unless they appear inside
                an object as a nested string. Each resulting string ends with the
                last char removed from all.

            """
            return "".join(
                [indent + l + "\n" for e in v for l in str(e).splitlines()]
            ).rstrip()

        def replace(match: Match[str]) -> str:
            """
            This function takes a Match[str] object as input and returns a string.
            It's purpose is to format strings that are delimited by "[" and "]"
            using indentations based on the nesting level of the "{" and "}"
            brackets. The function first extracts the indent (leading space) and
            the key (string between braces). It then uses the lookup() function
            to get a list of values for the key or a single value if it is not a
            list. Finally it formats the list/values using comma separating each
            item and returns the formatted string.

            Args:
                match (Match[str]): Here's an explanation of how match is used
                    within the replace() function:
                    
                    It uses re patterns for parameterized replacements with a named
                    group for the indentation length followed by the key as captured
                    groups. In simpler terms 'match' works just like a flag to
                    specify when searching patterns and retrieving sub-expressions/groups
                    of patterns that have been found.
                    Inside the code snippet shared previously we passed a match
                    parameter as input to the replace() function so therefore
                    "match" acts as the result object for each substitution. To
                    refer back to the sub expressions or the captured groups from
                    patterns (if present) when replacing we need to extract
                    properties using integer indexed keys off 'match', for example
                    .group(x). Note that match provides named and numbered access
                    to parts of a pattern matching expression.
                    The code works with: "indent" == match.group(1) key ==
                    match.group(2), where 'match.group(1)' matches the group one
                    capture for patterns played during the search process of this
                    string ,and 'key' will return match.group(2), or match expression
                    for this case - the 2nd capture of what the first (the leading
                    whitespace) and second capturing groups which were matched successfully
                    That is how 'match' function was put to use as an input parameter
                    with some specific functionalities described within that
                    Replace() code block snippet shared here earlier

            Returns:
                str: This function takes a string as input and returns an indented
                string. Specifically it returns a list of strings joined together
                with a comma separator if the input is a list. Otherwise it returns
                the string itself with no indentation.
                
                The output returned is a str

            """
            indent = match.group(1)
            key = match.group(2)
            comma_before = ""
            comma_after = ""
            if key[0] == "{":
                key = key[1:-1]
                if key[0] == ",":
                    comma_before = ", "
                    key = key[1:]
                if key[-1] == ",":
                    comma_after = ", "
                    key = key[:-1]
            v = lookup(key)
            if indent is not None:
                if not isinstance(v, list):
                    v = [v]
                return indent_lines(indent, v)
            elif isinstance(v, list):
                middle = ", ".join([str(x) for x in v])
                if len(v) == 0:
                    return middle
                return comma_before + middle + comma_after
            else:
                return str(v)

        return self.substitution.sub(replace, self.pattern)


if __name__ == "__main__":
    c = CodeTemplate(
        """\
    int foo($args) {

        $bar
            $bar
        $a+$b
    }
    int commatest(int a${,stuff})
    int notest(int a${,empty,})
    """
    )
    print(
        c.substitute(
            args=["hi", 8],
            bar=["what", 7],
            a=3,
            b=4,
            stuff=["things...", "others"],
            empty=[],
        )
    )
