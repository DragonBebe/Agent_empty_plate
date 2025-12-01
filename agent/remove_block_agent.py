"""High-level agent orchestration for removing the block from the plate using SmolVLA policies.

This module encapsulates the logic required to:

- Control the MuJoCo simulation environment
- Capture RGB observations for both policy inference and VLM inspection
- Call a Vision-Language Model (VLM) such as GPT for scene understanding and outcome verification
- Dynamically switch between multiple SmolVLA policies according to the VLM feedback

The code is intentionally structured to separate core responsibilities so that the
Jupyter workflow can remain concise while still being configurable and debuggable.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import textwrap
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    import requests
except ImportError:  # pragma: no cover - requests is an optional dependency during dry runs
    requests = None

from mujoco_env.y_env2_removeBlock_Pnp_mug import SimpleEnvRemoveBlock_pnp_mug

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses and configuration containers
# ---------------------------------------------------------------------------


@dataclass
class VLMSceneDescription:
    """Structured output from the VLM after analysing the initial scene."""

    task_summary: str
    block_region: str
    banana_region: Optional[str] = None
    notes: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLMOutcomeAssessment:
    """Structured output from the VLM after the policy attempt."""

    success: bool
    reason: str
    block_region: Optional[str] = None
    suggested_policy: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RemoveBlockAgentConfig:
    """Configuration bundle for the remove-block agent."""

    xml_scene_path: Optional[str] = None
    dataset_root: str = "./demo_data_language"
    dataset_repo_id: str = "omy_pnp_language"
    policy_repos: Dict[str, str] = field(
        default_factory=lambda: {
            "center": "DragonHu/smolvla_UR52_empty_plat_block_fixed_v2",
            # Fill in the variants below when dedicated checkpoints are available:
            # "left": "<your-huggingface-repo-left>",
            # "right": "<your-huggingface-repo-right>",
            # "front": "<your-huggingface-repo-front>",
            # "back": "<your-huggingface-repo-back>",
        }
    )
    default_policy: str = "center"
    robot_type: str = "ur"  # Supported: "ur", "so100"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    device: str = "cuda"
    image_size: int = 256
    sim_steps_per_action: int = 8
    max_steps_per_attempt: int = 180
    render: bool = True
    log_interval: int = 20
    save_debug_images: bool = False
    debug_dir: str = "./media/remove_block_agent"
    seed: Optional[int] = None
    vlm_model: str = "gpt-4.1-mini"
    vlm_base_url: str = "https://api.openai.com/v1"
    vlm_dry_run: bool = True

    def ensure(self) -> None:
        """Make sure configuration directories exist when needed."""

        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# VLM client
# ---------------------------------------------------------------------------


class GPTVLMClient:
    """Thin wrapper around OpenAI-style VLM endpoints.

    The client can operate in dry-run mode, producing deterministic mock outputs
    that unblock local development when API access is not yet configured.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        dry_run: bool = True,
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.dry_run = dry_run or (self.api_key is None)
        self.timeout = timeout

        if self.dry_run:
            LOGGER.info("GPTVLMClient operating in dry-run mode (no API calls will be made).")
        else:
            if requests is None:
                raise ImportError("`requests` package is required for live VLM calls.")

    # ------------------------------------------------------------------
    # Public high-level helpers
    # ------------------------------------------------------------------

    def describe_scene(self, image_bytes: bytes, task_hint: str) -> VLMSceneDescription:
        """Ask the VLM to describe the initial scene and extract high-level cues."""

        prompt = textwrap.dedent(
            f"""
            You are assisting a robotic manipulation task. Analyse the provided image and
            summarise the scene succinctly. Identify the red block relative to the plate
            (left/right/front/back/center) and optionally locate the banana if visible.
            The high-level goal is: {task_hint}

            Respond strictly in JSON with the fields:
            - task_summary: short natural language description
            - block_region: one of [left, right, front, back, center, unknown]
            - banana_region: optional string if a banana is visible (same categories)
            - notes: optional additional remarks
            """
        ).strip()

        payload = self._build_payload(prompt, image_bytes)
        raw = self._dispatch(payload)
        parsed = self._extract_json(raw)

        return VLMSceneDescription(
            task_summary=parsed.get("task_summary", ""),
            block_region=parsed.get("block_region", "unknown"),
            banana_region=parsed.get("banana_region"),
            notes=parsed.get("notes"),
            raw=raw,
        )

    def evaluate_outcome(self, image_bytes: bytes, stage_hint: str) -> VLMOutcomeAssessment:
        """Request the VLM to judge whether the stage succeeded and diagnose failures."""

        prompt = textwrap.dedent(
            f"""
            Inspect the image after the policy attempt for stage: {stage_hint}.
            Determine if the red block has been successfully removed from the plate.
            If it failed, explain the most likely reason and estimate where the block
            currently sits relative to the plate.

            Respond strictly in JSON with the fields:
            - success: boolean
            - reason: short natural language explanation
            - block_region: optional string [left, right, front, back, center, unknown]
            - suggested_policy: optional name for a better policy choice
            """
        ).strip()

        payload = self._build_payload(prompt, image_bytes)
        raw = self._dispatch(payload)
        parsed = self._extract_json(raw)

        return VLMOutcomeAssessment(
            success=bool(parsed.get("success", False)),
            reason=parsed.get("reason", ""),
            block_region=parsed.get("block_region"),
            suggested_policy=parsed.get("suggested_policy"),
            raw=raw,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_payload(self, prompt: str, image_bytes: bytes) -> Dict[str, Any]:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful robotics assistant that answers in JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }

    def _dispatch(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.dry_run:
            # Provide deterministic mock outputs for offline development
            return {
                "mock": True,
                "payload": payload,
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "task_summary": "模拟输出 (dry-run)",
                                    "block_region": "center",
                                    "banana_region": None,
                                    "notes": "使用 dry-run 模式，无真实 API 调用。",
                                    "success": False,
                                    "reason": "dry-run",
                                }
                            )
                        }
                    }
                ],
            }

        if requests is None:
            raise RuntimeError("Cannot perform live VLM call because `requests` is unavailable.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.base_url}/chat/completions"
        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _extract_json(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        try:
            choices = raw.get("choices", [])
            if not choices:
                return {}
            message = choices[0].get("message", {})
            content = message.get("content")
            if not content:
                return {}
            if isinstance(content, list):
                # Newer API variants wrap tool outputs in a list
                content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
            return json.loads(content)
        except Exception as exc:  # pragma: no cover - defensive parsing
            LOGGER.warning("Failed to parse VLM JSON output: %s", exc)
            return {}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def image_array_to_png_bytes(array: np.ndarray) -> bytes:
    """Convert an RGB numpy array to PNG bytes."""

    image = Image.fromarray(array)
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def ensure_tensor(data: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to the desired device if needed."""

    if data.device != device:
        return data.to(device)
    return data


# ---------------------------------------------------------------------------
# Agent implementation
# ---------------------------------------------------------------------------


class RemoveBlockAgent:
    """Coordinator that links MuJoCo simulation, SmolVLA policies, and the VLM."""

    def __init__(
        self,
        config: RemoveBlockAgentConfig,
        env: Optional[SimpleEnvRemoveBlock_pnp_mug] = None,
        vlm_client: Optional[GPTVLMClient] = None,
    ) -> None:
        self.config = config
        self.config.ensure()

        self.device = torch.device(config.device)
        self.env = env or self._create_environment()
        self.vlm = vlm_client or GPTVLMClient(
            model_name=config.vlm_model,
            base_url=config.vlm_base_url,
            dry_run=config.vlm_dry_run,
        )

        self.dataset_metadata = self._load_dataset_metadata()
        self.policy_cache: Dict[str, SmolVLAPolicy] = {}

        self.transform = transforms.Compose([transforms.ToTensor()])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_episode(self, instruction: Optional[str] = None, max_attempts: Optional[int] = None) -> Dict[str, Any]:
        """Execute the agent pipeline for a single episode."""

        summary: Dict[str, Any] = {
            "attempts": [],
            "initial_scene": None,
            "success": False,
        }

        attempts = max_attempts or len(self.config.policy_repos)
        attempts = max(attempts, 1)

        for attempt_idx in range(attempts):
            LOGGER.info("Starting attempt %d", attempt_idx + 1)
            self.env.reset(seed=self.config.seed)

            if instruction:
                self.env.set_instruction(instruction)

            agent_rgb, _ = self.env.grab_image()
            initial_scene = self.vlm.describe_scene(
                image_array_to_png_bytes(agent_rgb), self.env.instruction
            )

            if attempt_idx == 0:
                summary["initial_scene"] = initial_scene.raw

            policy_key = self._choose_policy(initial_scene, attempt_idx)
            policy = self._get_policy(policy_key)
            policy.reset()
            LOGGER.info("Selected policy '%s' for attempt %d", policy_key, attempt_idx + 1)

            attempt_record = {
                "policy": policy_key,
                "initial_scene": initial_scene.raw,
                "actions": 0,
                "vlm_outcome": None,
            }

            rollout_result = self._execute_policy(policy)
            attempt_record["actions"] = rollout_result["steps"]

            agent_rgb_after, _ = self.env.grab_image()
            outcome = self.vlm.evaluate_outcome(
                image_array_to_png_bytes(agent_rgb_after), stage_hint="remove_red_block"
            )
            attempt_record["vlm_outcome"] = outcome.raw

            summary["attempts"].append(attempt_record)

            if outcome.success:
                summary["success"] = True
                LOGGER.info("Attempt %d succeeded.", attempt_idx + 1)
                break

            LOGGER.info("Attempt %d failed: %s", attempt_idx + 1, outcome.reason)

        return summary

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_environment(self):
        robot = (self.config.robot_type or "ur").lower()

        if robot == "so100":
            from mujoco_env.y_env2_removeBlock_Test_random import EnvRemoveBlockTestRandom

            xml_path = self.config.xml_scene_path or "./asset/scene_remove_block_so100.xml"
            kwargs = {
                "action_type": "joint_angle",
                "state_type": "joint_angle",
            }
            kwargs.update(self.config.env_kwargs)
            return EnvRemoveBlockTestRandom(
                xml_path,
                seed=self.config.seed,
                **kwargs,
            )

        # default to UR robot environment
        xml_path = self.config.xml_scene_path or "./asset/scene_remove_block.xml"
        kwargs = {"action_type": "joint_angle"}
        kwargs.update(self.config.env_kwargs)
        return SimpleEnvRemoveBlock_pnp_mug(
            xml_path,
            seed=self.config.seed,
            **kwargs,
        )

    def _load_dataset_metadata(self) -> LeRobotDatasetMetadata:
        try:
            return LeRobotDatasetMetadata(self.config.dataset_repo_id, root=self.config.dataset_root)
        except FileNotFoundError:
            # Fallback if dataset root differs (e.g., cloned repository)
            LOGGER.warning(
                "Dataset '%s' not found under root '%s'. Attempting fallback './omy_pnp_language'.",
                self.config.dataset_repo_id,
                self.config.dataset_root,
            )
            return LeRobotDatasetMetadata(self.config.dataset_repo_id, root="./omy_pnp_language")

    def _choose_policy(self, scene: VLMSceneDescription, attempt_idx: int) -> str:
        if attempt_idx > 0:
            # After the first attempt, try to honour the region suggestion when available
            if scene.block_region and scene.block_region in self.config.policy_repos:
                return scene.block_region

        # Default routing: prioritise scene hint when possible, otherwise fallback
        region = scene.block_region or self.config.default_policy
        if region in self.config.policy_repos:
            return region
        LOGGER.info(
            "Region '%s' not registered, using default policy '%s'.",
            region,
            self.config.default_policy,
        )
        return self.config.default_policy

    def _get_policy(self, key: str) -> SmolVLAPolicy:
        if key not in self.config.policy_repos:
            raise KeyError(f"Policy key '{key}' is not registered in config.policy_repos")

        repo_id = self.config.policy_repos[key]
        if repo_id in self.policy_cache:
            return self.policy_cache[repo_id]

        LOGGER.info("Loading SmolVLA policy from %s", repo_id)
        policy = SmolVLAPolicy.from_pretrained(repo_id, dataset_stats=self.dataset_metadata.stats)
        policy.to(self.device)
        policy.eval()
        self.policy_cache[repo_id] = policy
        return policy

    def _execute_policy(self, policy: SmolVLAPolicy) -> Dict[str, Any]:
        instruction = self.env.instruction
        steps = 0
        horizon = self.config.max_steps_per_attempt

        for step_idx in range(horizon):
            data = self._build_policy_input(instruction)
            action = policy.select_action(data)
            action = action[0, :7].detach().cpu().numpy()

            self.env.step(action)
            for _ in range(self.config.sim_steps_per_action):
                self.env.step_env()

            if self.config.render:
                self.env.render()

            steps += 1

            if steps % self.config.log_interval == 0:
                LOGGER.info("Step %d/%d completed for current attempt.", steps, horizon)

            if self.env.check_success():
                LOGGER.info("Environment reported success after %d steps.", steps)
                break

        return {"steps": steps}

    def _build_policy_input(self, instruction: str) -> Dict[str, Any]:
        joint_state = np.asarray(self.env.get_joint_state()[:6], dtype=np.float32)
        state = torch.from_numpy(joint_state).unsqueeze(0)
        rgb_agent, rgb_wrist = self.env.grab_image()

        main_image = self._process_image(rgb_agent)
        wrist_image = self._process_image(rgb_wrist)

        data = {
            "observation.state": ensure_tensor(state, self.device),
            "observation.image": ensure_tensor(main_image.unsqueeze(0), self.device),
            "observation.wrist_image": ensure_tensor(wrist_image.unsqueeze(0), self.device),
            "task": [instruction],
        }

        return data

    def _process_image(self, array: np.ndarray) -> torch.Tensor:
        image = Image.fromarray(array).resize((self.config.image_size, self.config.image_size))
        tensor = self.transform(image)
        return tensor


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_default_agent(
    config: Optional[RemoveBlockAgentConfig] = None,
) -> RemoveBlockAgent:
    """Instantiate a ready-to-use agent with the default configuration."""

    config = config or RemoveBlockAgentConfig()
    return RemoveBlockAgent(config=config)


__all__ = [
    "RemoveBlockAgent",
    "RemoveBlockAgentConfig",
    "GPTVLMClient",
    "VLMSceneDescription",
    "VLMOutcomeAssessment",
    "create_default_agent",
]


