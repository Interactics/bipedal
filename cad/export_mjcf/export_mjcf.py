"""
export_mjcf.py - Export the active OP3_sim design to MJCF with ACDC4Robot, without dialogs.

Needs the ACDC4Robot add-in (v1.1.1) RUNNING in Fusion (it provides the modules).
Run with OP3_sim as the active, saved document. Writes to OUT_DIR/<design name>/:
  <name>.xml, meshes/*.stl, acdc4robot-export-report.json
"""
import adsk.core, adsk.fusion, os, sys, traceback

OUT_DIR = '/Users/hoyeon_yu/Documents/private_ws/bipedal/robot_sim/model'
EXPECTED_DOC = 'OP3_sim'


def _acdc_modules():
    mods = {}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None) or ''
        if '/ACDC4Robot/' in f:
            mods[f.split('/ACDC4Robot/', 1)[1]] = m
    need = ['core/preflight.py', 'core/utils.py', 'core/write.py',
            'commands/ACDC4Robot/acdc4robot.py', 'commands/ACDC4Robot/constants.py']
    missing = [n for n in need if n not in mods]
    if missing:
        raise RuntimeError('ACDC4Robot add-in is not running (missing %s). '
                           'Start it in Scripts and Add-Ins.' % missing)
    return [mods[n] for n in need]


def export():
    app = adsk.core.Application.get()
    if app.activeDocument.name != EXPECTED_DOC:
        raise RuntimeError('Make %s the active document (active: %s)'
                           % (EXPECTED_DOC, app.activeDocument.name))
    pre, utils, write, cmd, const = _acdc_modules()
    design = adsk.fusion.Design.cast(app.activeProduct)
    root = design.rootComponent
    name = utils.get_valid_filename(root.name) or 'robot'
    const.set_robot_name(name)
    const.set_rdf('MJCF')
    for fn in ('set_sim_env', 'set_simulator', 'set_sim'):
        if hasattr(const, fn):
            getattr(const, fn)('MuJoCo')
            break
    rep = pre.inspect_mjcf_design(design)
    if not rep['passed']:
        raise RuntimeError(pre.format_report(rep))
    out = os.path.join(OUT_DIR, name)
    os.makedirs(out, exist_ok=True)
    pre.write_report(out, rep)
    links, joints = cmd.get_link_joint_list(design)
    write.write_mjcf(root, name, out)
    cmd.export_stl(design, out, links)
    return 'Exported %s to\n%s\n\n%s' % (name, out, pre.format_report(rep))


def run(context):
    ui = adsk.core.Application.get().userInterface
    try:
        ui.messageBox(export(), 'export_mjcf')
    except Exception:
        ui.messageBox('Failed:\n' + traceback.format_exc(), 'export_mjcf')
