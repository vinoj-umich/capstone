# Cache class to store models
class ModelCache:
    """
    This class provides caching functionality for storing and retrieving models.
    It ensures that a model is loaded only once during the session.
    """
    def __init__(self):
        self.cache = {}

    def get(self, model_id):
        """
        Retrieve a cached model by its ID.

        Args:
        - model_id (str): The ID of the model to retrieve.

        Returns:
        - ModelQA: The cached model instance, or None if not found.
        """
        return self.cache.get(model_id)

    def set(self, model_id, model_qa):
        """
        Cache a model instance.

        Args:
        - model_id (str): The ID of the model.
        - model_qa (ModelQA): The ModelQA instance to cache.
        """
        self.cache[model_id] = model_qa
