"""Tests for Slakh patch selection stub."""

from synthesis.patches import PatchAssignment, select_patch


def test_select_patch_passthrough():
    assignment = select_patch(program=11, is_drum=False)
    assert assignment == PatchAssignment(program=11, is_drum=False)


def test_select_patch_drum():
    assignment = select_patch(program=0, is_drum=True)
    assert assignment.is_drum is True
