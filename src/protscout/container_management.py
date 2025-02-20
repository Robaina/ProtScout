from __future__ import annotations

import subprocess
import logging
from typing import List, Dict, Tuple, Optional, Any
import atexit


class DockerContainerPool:
    """
    Singleton class that maintains long-running Docker containers for reuse
    across multiple prediction runs.
    """

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DockerContainerPool()
            # Register cleanup on exit
            atexit.register(cls._instance.cleanup)
        return cls._instance

    def __init__(self):
        self.containers = {}  # key -> container_id
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing DockerContainerPool")

    def get_container(
        self,
        image_name: str,
        volume_mounts: Optional[List[Tuple[str, str]]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        gpu: bool = False,
        network: Optional[str] = None,
    ) -> str:
        """
        Get or create a Docker container with the specified configuration.

        Args:
            image_name (str): Docker image name to use
            volume_mounts (list): List of (source, destination) volume mount tuples
            env_vars (dict): Dictionary of environment variables to set
            gpu (bool): Whether GPU access is required
            network (str): Docker network to connect the container to

        Returns:
            str: Container ID of running container
        """
        # Create a unique key based on configuration
        key_parts = [image_name]
        if volume_mounts:
            key_parts.extend(sorted(f"{src}:{dst}" for src, dst in volume_mounts))
        if env_vars:
            key_parts.extend(sorted(f"{k}={v}" for k, v in env_vars.items()))
        if gpu:
            key_parts.append("gpu=true")
        if network:
            key_parts.append(f"network={network}")

        key = ":".join(key_parts)

        # Check if we already have a container with this configuration
        if key in self.containers:
            container_id = self.containers[key]
            if self._is_container_running(container_id):
                self.logger.debug(f"Reusing existing container: {container_id[:12]}")
                return container_id
            # Container stopped - remove reference
            self.logger.warning(
                f"Container {container_id[:12]} stopped unexpectedly, creating new one"
            )
            del self.containers[key]

        # Create a new container
        container_id = self._create_container(
            image_name, volume_mounts, env_vars, gpu, network
        )
        self.containers[key] = container_id
        self.logger.info(f"Created new container: {container_id[:12]}")
        return container_id

    def _is_container_running(self, container_id: str) -> bool:
        """
        Check if a container is still running

        Args:
            container_id (str): Container ID to check

        Returns:
            bool: True if container is running, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode == 0 and "true" in result.stdout.strip()
        except Exception as e:
            self.logger.error(f"Error checking container status: {e}")
            return False

    def _create_container(
        self,
        image_name: str,
        volume_mounts: Optional[List[Tuple[str, str]]] = None,
        env_vars: Optional[Dict[str, str]] = None,
        gpu: bool = False,
        network: Optional[str] = None,
    ) -> str:
        """
        Create a new long-running container

        Args:
            image_name (str): Docker image name
            volume_mounts (list): Volume mounts as (source, destination) tuples
            env_vars (dict): Environment variables
            gpu (bool): Whether to enable GPU
            network (str): Docker network to connect to

        Returns:
            str: Container ID of the new container
        """
        cmd = ["docker", "run", "-d"]

        # Add volume mounts
        if volume_mounts:
            for src, dst in volume_mounts:
                cmd.extend(["-v", f"{src}:{dst}"])

        # Add environment variables
        if env_vars:
            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])

        # Add GPU if needed
        if gpu:
            cmd.extend(["--gpus", "all"])

        # Add network if specified
        if network:
            cmd.extend(["--network", network])

        # Add image and keep-alive command (tail -f keeps container running)
        cmd.extend([image_name, "tail", "-f", "/dev/null"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create container: {e.stderr}")
            raise RuntimeError(f"Failed to create Docker container: {e.stderr}")

    def execute_command(
        self,
        container_id: str,
        command: List[str],
        capture_output: bool = True,
        workdir: Optional[str] = None,
    ) -> subprocess.CompletedProcess:
        """
        Execute a command in an existing container

        Args:
            container_id (str): Container ID
            command (list): Command to execute as a list of strings
            capture_output (bool): Whether to capture and return command output
            workdir (str): Working directory in the container

        Returns:
            CompletedProcess: Result of command execution
        """
        cmd = ["docker", "exec"]
        if capture_output:
            cmd.append("-i")  # Interactive mode
        if workdir:
            cmd.extend(["-w", workdir])
        cmd.append(container_id)
        cmd.extend(command)

        self.logger.debug(
            f"Executing in container {container_id[:12]}: {' '.join(command[:3])}..."
        )

        try:
            return subprocess.run(
                cmd, capture_output=capture_output, text=True, check=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Command failed in container {container_id[:12]}: {e.stderr}"
            )
            raise

    def cleanup(self) -> None:
        """Stop and remove all containers in the pool"""
        self.logger.info(f"Cleaning up {len(self.containers)} containers")
        for container_id in list(self.containers.values()):
            try:
                subprocess.run(
                    ["docker", "stop", container_id], capture_output=True, check=False
                )
                subprocess.run(
                    ["docker", "rm", container_id], capture_output=True, check=False
                )
                self.logger.debug(f"Removed container: {container_id[:12]}")
            except Exception as e:
                self.logger.error(f"Error during container cleanup: {e}")
        self.containers.clear()
