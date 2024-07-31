import logging
import os
from typing import Any, Dict, Optional, Union
import autogen as orgautogen
from config import OPENAI_API_KEY_DICT, TMPPATH
from autogen.cache.abstract_cache_base import AbstractCache
from autogen.cache.disk_cache import DiskCache
orgautogen.oai.client.LEGACY_CACHE_DIR = TMPPATH
##########################################################################
#rewrite CachFactory for deploying on aws lambda which only allow to write in /tmp folder
##########################################################################
class CustomedAutogenCacheFactory:
    @staticmethod
    def cache_factory(
        seed: Union[str, int],
        redis_url: Optional[str] = None,
        cache_path_root: str = ".cache",
        cosmosdb_config: Optional[Dict[str, Any]] = None,
    ) -> AbstractCache:
        """
        Factory function for creating cache instances.

        This function decides whether to create a RedisCache, DiskCache, or CosmosDBCache instance
        based on the provided parameters. If RedisCache is available and a redis_url is provided,
        a RedisCache instance is created. If connection_string, database_id, and container_id
        are provided, a CosmosDBCache is created. Otherwise, a DiskCache instance is used.

        Args:
            seed (Union[str, int]): Used as a seed or namespace for the cache.
            redis_url (Optional[str]): URL for the Redis server.
            cache_path_root (str): Root path for the disk cache.
            cosmosdb_config (Optional[Dict[str, str]]): Dictionary containing 'connection_string',
                                                       'database_id', and 'container_id' for Cosmos DB cache.

        Returns:
            An instance of RedisCache, DiskCache, or CosmosDBCache.

        Examples:

        Creating a Redis cache

        ```python
        redis_cache = cache_factory("myseed", "redis://localhost:6379/0")
        ```
        Creating a Disk cache

        ```python
        disk_cache = cache_factory("myseed", None)
        ```

        Creating a Cosmos DB cache:
        ```python
        cosmos_cache = cache_factory("myseed", cosmosdb_config={
                "connection_string": "your_connection_string",
                "database_id": "your_database_id",
                "container_id": "your_container_id"}
            )
        ```

        """
        if redis_url:
            try:
                from autogen.cache.redis_cache import RedisCache

                return RedisCache(seed, redis_url)
            except ImportError:
                logging.warning(
                    "RedisCache is not available. Checking other cache options. The last fallback is DiskCache."
                )

        if cosmosdb_config:
            try:
                from autogen.cache.cosmos_db_cache import CosmosDBCache

                return CosmosDBCache.create_cache(seed, cosmosdb_config)

            except ImportError:
                logging.warning("CosmosDBCache is not available. Fallback to DiskCache.")

        # Default to DiskCache if neither Redis nor Cosmos DB configurations are provided
        path = os.path.join(cache_path_root, str(seed))
        if path.find("/tmp") ==  0: #code for aws lambda
          new_path = os.path.join(path)
        else: new_path = os.path.join(".", path) #original code
        print("[LOG-Autogen-byDucAnh]--> create autogencache at ", new_path)
        return DiskCache(new_path)

orgautogen.cache.cache_factory.CacheFactory = CustomedAutogenCacheFactory
##############################################################################
#                      ------------END---------
###############################################################################

