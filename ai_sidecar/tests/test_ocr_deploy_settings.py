"""
Device-free checks that the shipped OCR deploy settings point at the HTTP sidecar.

Run from the repository root: python -m pytest ai_sidecar/tests/test_ocr_deploy_settings.py
"""
import ast
import shutil
from pathlib import Path

import pytest

# Importing deploy.logger changes the working directory to the repository root,
# so these imports must stay above any chdir done by a test.
import deploy.Windows.config as windows_config
import deploy.config as deploy_config
from deploy.utils import poor_yaml_read, poor_yaml_write

ROOT = Path(__file__).resolve().parents[2]
SIDECAR_ADDRESS = '127.0.0.1:8484'
LEGACY_ADDRESS = '127.0.0.1:22268'

CONFIG_TEMPLATES = sorted(p.name for p in (ROOT / 'config').glob('deploy.template*.yaml'))
DOCKER_TEMPLATES = [name for name in CONFIG_TEMPLATES if 'docker' in name]
HOST_TEMPLATES = [name for name in CONFIG_TEMPLATES if 'docker' not in name]


def read_template(relative):
    # The same parser DeployConfig.read() uses, so the test sees what ALAS sees.
    return poor_yaml_read(str(ROOT / relative))


def test_every_template_is_covered():
    assert len(DOCKER_TEMPLATES) == 2
    assert len(HOST_TEMPLATES) == 6


# ---------------------------------------------------------------------------
# Templates agree with the class defaults
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('relative, model', [
    ('deploy/template', deploy_config.ConfigModel),
    ('deploy/Windows/template.yaml', windows_config.ConfigModel),
] + [('config/' + name, deploy_config.ConfigModel) for name in HOST_TEMPLATES])
def test_template_agrees_with_class_defaults(relative, model):
    assert model.UseOcrServer is True
    assert model.OcrClientAddress == SIDECAR_ADDRESS

    data = read_template(relative)
    assert data['UseOcrServer'] is model.UseOcrServer
    assert data['OcrClientAddress'] == model.OcrClientAddress
    assert data['StartOcrServer'] is model.StartOcrServer
    assert data['OcrServerPort'] == model.OcrServerPort


@pytest.mark.parametrize('name', DOCKER_TEMPLATES)
def test_docker_template_address_exists_in_compose(name):
    """The address must name something docker-compose.yml lets the alas service reach."""
    yaml = pytest.importorskip('yaml')
    data = read_template('config/' + name)
    assert data['UseOcrServer'] is True
    assert data['StartOcrServer'] is False
    host, _, port = data['OcrClientAddress'].rpartition(':')
    assert port == '8484'

    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text(encoding='utf-8'))
    services = compose['services']
    extra_hosts = [entry.split(':')[0] for entry in services['alas'].get('extra_hosts', [])]
    assert host in list(services) + extra_hosts


def test_committed_templates_are_generator_output(tmp_path, monkeypatch):
    """config/deploy.template*.yaml are generated from deploy/template, never hand-edited."""
    from module.config.config_updater import ConfigGenerator

    (tmp_path / 'deploy').mkdir()
    (tmp_path / 'config').mkdir()
    shutil.copy(str(ROOT / 'deploy/template'), str(tmp_path / 'deploy/template'))
    monkeypatch.chdir(tmp_path)
    ConfigGenerator.generate_deploy_template()

    generated = sorted(p.name for p in (tmp_path / 'config').glob('deploy.template*.yaml'))
    assert generated == CONFIG_TEMPLATES
    for name in CONFIG_TEMPLATES:
        expected = (tmp_path / 'config' / name).read_text(encoding='utf-8')
        committed = (ROOT / 'config' / name).read_text(encoding='utf-8')
        assert committed == expected, name


# ---------------------------------------------------------------------------
# What DeployConfig.read() does to a user's config/deploy.yaml
# ---------------------------------------------------------------------------

@pytest.fixture
def install(tmp_path, monkeypatch):
    (tmp_path / 'deploy').mkdir()
    (tmp_path / 'config').mkdir()
    shutil.copy(str(ROOT / 'deploy/template'), str(tmp_path / 'deploy/template'))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_fresh_install_uses_the_sidecar(install):
    config = deploy_config.DeployConfig()
    assert config.UseOcrServer is True
    assert config.OcrClientAddress == SIDECAR_ADDRESS

    written = poor_yaml_read('./config/deploy.yaml')
    assert written['UseOcrServer'] is True
    assert written['OcrClientAddress'] == SIDECAR_ADDRESS


def test_existing_install_keeps_its_ocr_settings(install):
    """Values already in deploy.yaml win over the template; nothing is migrated."""
    legacy = poor_yaml_read('./deploy/template')
    legacy.update({'UseOcrServer': False, 'OcrClientAddress': LEGACY_ADDRESS})
    poor_yaml_write(legacy, './config/deploy.yaml')

    config = deploy_config.DeployConfig()
    assert config.UseOcrServer is False
    assert config.OcrClientAddress == LEGACY_ADDRESS

    written = poor_yaml_read('./config/deploy.yaml')
    assert written['UseOcrServer'] is False
    assert written['OcrClientAddress'] == LEGACY_ADDRESS


def test_missing_ocr_keys_are_filled_from_the_template(install):
    (install / 'config/deploy.yaml').write_text('Deploy:\n  Git:\n    Branch: dev\n', encoding='utf-8')

    config = deploy_config.DeployConfig()
    assert config.Branch == 'dev'
    assert config.UseOcrServer is True
    assert config.OcrClientAddress == SIDECAR_ADDRESS

    written = poor_yaml_read('./config/deploy.yaml')
    assert written['Branch'] == 'dev'
    assert written['OcrClientAddress'] == SIDECAR_ADDRESS


# ---------------------------------------------------------------------------
# Legacy ZeroRPC server gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('start_ocr_server, use_ocr_server, expected', [
    (False, False, False),
    (False, True, False),
    (True, False, True),
    (True, True, False),
])
def test_legacy_ocr_server_gate(start_ocr_server, use_ocr_server, expected):
    from module.ocr.rpc import should_start_legacy_ocr_server

    assert should_start_legacy_ocr_server(start_ocr_server, use_ocr_server) is expected


# ---------------------------------------------------------------------------
# cnocr stays out of module import time
# ---------------------------------------------------------------------------

def is_type_checking_guard(node):
    test = node.test
    return (isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING') \
        or (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING')


def import_time_modules(body):
    """Yield modules imported when a file is imported, descending into module-level blocks."""
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module
        elif isinstance(node, ast.If):
            if not is_type_checking_guard(node):
                for name in import_time_modules(node.body):
                    yield name
            for name in import_time_modules(node.orelse):
                yield name
        elif isinstance(node, ast.Try):
            for block in [node.body, node.orelse, node.finalbody] + [h.body for h in node.handlers]:
                for name in import_time_modules(block):
                    yield name


def test_nothing_imports_cnocr_at_import_time():
    """al_ocr pulls in cnocr and mxnet, which an install using the HTTP sidecar lacks."""
    offenders = []
    for path in sorted((ROOT / 'module').rglob('*.py')):
        if path.name == 'al_ocr.py':
            continue
        source = path.read_text(encoding='utf-8')
        if 'al_ocr' not in source and 'cnocr' not in source:
            continue
        for name in import_time_modules(ast.parse(source).body):
            if name and (name == 'module.ocr.al_ocr' or name.split('.')[0] in ('cnocr', 'mxnet')):
                offenders.append('%s imports %s' % (path.relative_to(ROOT).as_posix(), name))
    assert offenders == []
