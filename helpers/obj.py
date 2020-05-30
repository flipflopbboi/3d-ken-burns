class HashableMixIn:
    def __init__(self, name: str):
        self.name = name

    def __lt__(self, other) -> bool:
        try:
            return self.name < other.id
        except Exception as err:
            if not isinstance(other, self.__class__):
                raise NotImplementedError(
                    f"Cannot compare {type(other)} with {type(self)}"
                )
            raise err

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.__class__.__name__}(id_={self.name})"
