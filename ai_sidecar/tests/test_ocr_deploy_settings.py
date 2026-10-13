"""
Device-free checks that the shipped OCR deploy settings and docker-compose.yml point at
the HTTP sidecar, and that the legacy ZeroRPC OCR server is gone.

Run from the repository root: python -m pytest ai_sidecar/tests/test_ocr_deploy_settings.py
"""
import ast
import re
import shutil
from pathlib import Path

import pytest

# Importing deploy.logger or module.logger changes the working directory to the
# repository root, so these imports must stay above any chdir done by a test.
import deploy.Windows.config as windows_config
import deploy.config as deploy_config
import module.webui.config as webui_config
from deploy.utils import poor_yaml_read, poor_yaml_write

ROOT = Path(__file__).resolve().parents[2]
SIDECAR_ADDRESS = '127.0.0.1:8484'
COMPOSE_ADDRESS = 'sidecar:8484'
LEGACY_ADDRESS = '127.0.0.1:22268'
SIDECAR_IMAGE = 'ghcr.io/aurablacklight/rust-gacha-ocr-sidecar:main'

CONFIG_TEMPLATES = sorted(p.name for p in (ROOT / 'config').glob('deploy.template*.yaml'))
DOCKER_TEMPLATES = [name for name in CONFIG_TEMPLATES if 'docker' in name]
HOST_TEMPLATES = [name for name in CONFIG_TEMPLATES if 'docker' not in name]
SOURCE_TEMPLATES = ['deploy/template', 'deploy/Windows/template.yaml']

# Deploy settings and functions that only served the ZeroRPC cnocr server.
REMOVED_SETTINGS = ('StartOcrServer', 'OcrServerPort')
REMOVED_NAMES = REMOVED_SETTINGS + (
    'start_ocr_server',
    'start_ocr_server_process',
    'stop_ocr_server_process',
    'should_start_legacy_ocr_server',
)


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


@pytest.mark.parametrize('name', DOCKER_TEMPLATES)
def test_docker_template_names_the_compose_sidecar(name, compose):
    """Inside the alas container 127.0.0.1 is the container itself, so the service is named."""
    data = read_template('config/' + name)
    assert data['UseOcrServer'] is True
    assert data['OcrClientAddress'] == COMPOSE_ADDRESS

    host, _, port = data['OcrClientAddress'].rpartition(':')
    assert host in compose['services']
    assert int(port) in [container_port(entry) for entry in compose['services'][host]['ports']]


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
# docker-compose.yml
# ---------------------------------------------------------------------------

def compose_text():
    return (ROOT / 'docker-compose.yml').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def compose():
    import yaml
    return yaml.safe_load(compose_text())


def container_port(entry):
    """Container side of a short-syntax ports entry such as 127.0.0.1:8484:8484."""
    return int(str(entry).split('/')[0].rpartition(':')[2])


def environment(service):
    """
    Returns:
        dict: Variable name to value. None for a bare NAME, which compose passes through.
    """
    out = {}
    for entry in service.get('environment', []):
        name, separator, value = entry.partition('=')
        out[name] = value if separator else None
    return out


def mounts(service):
    """
    Returns:
        dict: Container path to (source, mode), with ${NAME:-default} resolved to default.
    """
    out = {}
    for entry in service.get('volumes', []):
        parts = re.sub(r'\$\{\w+:-([^}]*)\}', r'\1', entry).split(':')
        out[parts[1]] = (parts[0], parts[2] if len(parts) > 2 else 'rw')
    return out


def toml_section(text, section):
    """Read `key = value` lines of one TOML table, without a TOML parser (Python 3.7)."""
    out = {}
    current = None
    for line in text.splitlines():
        line = line.split('#')[0].strip()
        if line.startswith('['):
            current = line.strip('[]')
        elif current == section and '=' in line:
            key, _, value = line.partition('=')
            out[key.strip()] = value.strip().strip('"')
    return out


def test_compose_runs_alas_and_the_sidecar_only(compose):
    assert sorted(compose['services']) == ['alas', 'sidecar']


def test_compose_has_no_legacy_ocr_service():
    text = compose_text().lower()
    for stale in ('dockerfile.ocr', '22268', 'alas-ocr', 'zerorpc'):
        assert stale not in text
    assert not (ROOT / 'deploy/docker/Dockerfile.ocr').exists()
    assert not (ROOT / 'deploy/docker/requirements-ocr.txt').exists()


def test_compose_sidecar_is_the_rust_image(compose):
    sidecar = compose['services']['sidecar']
    assert sidecar['image'] == SIDECAR_IMAGE
    # Not built from ./ai_sidecar, the Python sidecar without /ocr/recognize.
    assert 'build' not in sidecar
    assert 'ai_sidecar' not in compose_text()
    # The image is multi-arch, a pin would force emulation on the other architecture.
    assert 'platform' not in sidecar
    assert 8484 in [container_port(entry) for entry in sidecar['ports']]


def test_compose_sidecar_mounts(compose):
    volumes = mounts(compose['services']['sidecar'])

    source, mode = volumes['/app/config.toml']
    assert mode == 'ro'
    server = toml_section((ROOT / source).read_text(encoding='utf-8'), 'server')
    # 127.0.0.1, the server's default, cannot be reached from the alas container.
    assert server['host'] == '0.0.0.0'
    assert server['port'] == '8484'

    assert volumes['/app/models/ocr'][1] == 'ro'
    assert volumes['/app/state'][0] in compose['volumes']


def test_compose_sidecar_gets_the_provider_key(compose):
    env = environment(compose['services']['sidecar'])
    assert 'ANTHROPIC_API_KEY' in env
    # Variables of the Python sidecar, which gacha-server does not read.
    for stale in ('AI_API_KEY', 'AI_BASE_URL', 'AI_MODEL'):
        assert stale not in env


def test_compose_alas_waits_for_the_sidecar(compose):
    services = compose['services']
    assert services['alas']['depends_on']['sidecar']['condition'] == 'service_healthy'
    # The image has no curl. It is healthy by its own HEALTHCHECK unless compose overrides it.
    assert 'curl' not in str(services['sidecar'].get('healthcheck', ''))


def test_compose_alas_environment_is_read_by_alas(compose, monkeypatch):
    """No variable that looks like an address but that ALAS ignores."""
    from module.ai_hook.hook import AISidecarClient
    from module.config.config_generated import GeneratedConfig

    env = environment(compose['services']['alas'])
    assert 'OCR_SERVER_URL' not in env
    assert env['AI_SIDECAR_URL'] == 'http://' + COMPOSE_ADDRESS

    monkeypatch.setenv('AI_SIDECAR_URL', env['AI_SIDECAR_URL'])
    assert AISidecarClient(GeneratedConfig).url == 'http://' + COMPOSE_ADDRESS

    # Outside compose the variable is unset and the default is unchanged.
    monkeypatch.delenv('AI_SIDECAR_URL')
    assert GeneratedConfig.AiSidecar_SidecarUrl == 'http://' + SIDECAR_ADDRESS
    assert AISidecarClient(GeneratedConfig).url == 'http://' + SIDECAR_ADDRESS


# ---------------------------------------------------------------------------
# What DeployConfig.read() does to a user's config/deploy.yaml
# ---------------------------------------------------------------------------

@pytest.fixture
def install(tmp_path, monkeypatch):
    (tmp_path / 'deploy' / 'Windows').mkdir(parents=True)
    (tmp_path / 'config').mkdir()
    for relative in SOURCE_TEMPLATES:
        shutil.copy(str(ROOT / relative), str(tmp_path / relative))
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


# The installer, the Windows installer and the web UI each read config/deploy.yaml.
DEPLOY_CONFIGS = [
    (deploy_config, 'deploy/template'),
    (windows_config, 'deploy/Windows/template.yaml'),
    (webui_config, 'deploy/template'),
]


def write_legacy_install(template):
    """
    Write a config/deploy.yaml from before the ZeroRPC server was removed:
    in-process cnocr, and the two server settings that no longer exist.

    Returns:
        bytes: Content of the file.
    """
    legacy = poor_yaml_read(template)
    legacy.update({'UseOcrServer': False, 'OcrClientAddress': LEGACY_ADDRESS})
    poor_yaml_write(legacy, './config/deploy.yaml', template_file=template)

    file = Path('config/deploy.yaml')
    content, count = re.subn(
        br'(    UseOcrServer: false(\r?\n))',
        br'\1    StartOcrServer: true\2    OcrServerPort: 22268\2',
        file.read_bytes())
    assert count == 1
    file.write_bytes(content)
    return content


@pytest.mark.parametrize('module, template', DEPLOY_CONFIGS)
def test_existing_install_with_removed_settings_still_starts(install, module, template):
    """Unknown keys in an existing deploy.yaml are ignored, and the file is left alone."""
    before = write_legacy_install(template)

    config = module.DeployConfig()
    assert config.UseOcrServer is False
    assert config.OcrClientAddress == LEGACY_ADDRESS
    for name in REMOVED_SETTINGS:
        assert not hasattr(config, name)

    assert Path('config/deploy.yaml').read_bytes() == before


@pytest.mark.parametrize('module, template', DEPLOY_CONFIGS)
def test_rewritten_existing_install_keeps_its_ocr_values(install, module, template):
    """When ALAS does write the file, the removed settings drop out and the rest stays."""
    content = write_legacy_install(template)
    # A key missing from the file is what makes read() write it again.
    content, count = re.subn(br'    AutoConnect:.*\n', b'', content)
    assert count == 1
    Path('config/deploy.yaml').write_bytes(content)

    for _ in range(2):
        config = module.DeployConfig()
        assert config.UseOcrServer is False
        assert config.OcrClientAddress == LEGACY_ADDRESS

    written = poor_yaml_read('./config/deploy.yaml')
    assert 'AutoConnect' in written
    assert written['UseOcrServer'] is False
    assert written['OcrClientAddress'] == LEGACY_ADDRESS
    for name in REMOVED_SETTINGS:
        assert name not in written


# ---------------------------------------------------------------------------
# The legacy ZeroRPC OCR server is gone
# ---------------------------------------------------------------------------

def python_sources():
    """Yield (relative path, source) of the code ALAS runs, without the tests."""
    paths = sorted(ROOT.glob('*.py'))
    for folder in ('module', 'deploy', 'dev_tools', 'submodule'):
        paths += sorted((ROOT / folder).rglob('*.py'))
    for path in paths:
        yield path.relative_to(ROOT).as_posix(), path.read_text(encoding='utf-8')


def names_in(tree):
    """Yield every identifier, attribute, imported name and string literal in a module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.alias):
            yield node.name.split('.')[-1]
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.value
        elif type(node).__name__ == 'Str':
            # Python 3.7 parses string literals to ast.Str
            yield node.s


def imported_modules(tree):
    """Yield every module imported anywhere in a file, function bodies included."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


@pytest.mark.parametrize('relative', SOURCE_TEMPLATES + ['config/' + name for name in CONFIG_TEMPLATES])
def test_templates_have_no_legacy_server_settings(relative):
    text = (ROOT / relative).read_text(encoding='utf-8')
    for stale in REMOVED_SETTINGS + ('22268',):
        assert stale not in text


@pytest.mark.parametrize('model', [deploy_config.ConfigModel, windows_config.ConfigModel])
def test_config_models_have_no_legacy_server_settings(model):
    for name in REMOVED_SETTINGS:
        assert not hasattr(model, name)
    # These two select the HTTP sidecar and stay.
    assert hasattr(model, 'UseOcrServer')
    assert hasattr(model, 'OcrClientAddress')


def test_no_code_references_the_legacy_server():
    offenders = []
    for relative, source in python_sources():
        if not any(name in source for name in REMOVED_NAMES):
            continue
        found = set(names_in(ast.parse(source))) & set(REMOVED_NAMES)
        for name in sorted(found):
            offenders.append('%s references %s' % (relative, name))
    assert offenders == []


def test_no_code_imports_zerorpc():
    offenders = []
    for relative, source in python_sources():
        if 'zerorpc' not in source and 'zmq' not in source:
            continue
        for name in imported_modules(ast.parse(source)):
            if name.split('.')[0] in ('zerorpc', 'zmq'):
                offenders.append('%s imports %s' % (relative, name))
    assert offenders == []


def test_http_ocr_client_is_kept():
    import module.ocr.rpc as rpc

    assert callable(rpc.ModelProxy.ocr_for_single_lines)
    assert callable(rpc.ModelProxyFactory.close)
    for name in REMOVED_NAMES + ('alive', 'process'):
        assert not hasattr(rpc, name)


def test_in_process_cnocr_path_is_kept():
    """UseOcrServer: false still loads the cnocr models inside the alas instance."""
    assert (ROOT / 'module/ocr/models.py').exists()
    tree = ast.parse((ROOT / 'module/ocr/ocr.py').read_text(encoding='utf-8'))
    assert 'module.ocr.models' in set(import_time_modules(tree.body))
    assert 'module.ocr.rpc' in set(import_time_modules(tree.body))


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
