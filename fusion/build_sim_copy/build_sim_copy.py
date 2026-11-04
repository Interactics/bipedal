"""
build_sim_copy.py - Rebuild the flat OP3_sim design from Leg_ver1.

Run inside Fusion (UTILITIES > Scripts and Add-Ins) with Leg_ver1 as the ACTIVE document.
It creates a new, unsaved document. Save it as "OP3_sim" (replacing the old one),
then run export_mjcf with OP3_sim active.

What it does
  1. Sets all 12 joints of Leg_ver1 to 0 and checks the zero pose
     (printed parts must sit exactly on their hidden metal originals, which never move).
  2. Copies the visible solid bodies of each rigid group (= link) into one component per
     link, in a flat top-level structure (what ACDC4Robot's MJCF export needs).
     Each link's frame = frame of the XM430 that drives it (origin on the horn axis,
     +Z out of the horn). base_link keeps the world frame.
  3. Adds 12 as-built revolute joints, each about its child link's +Z.
  4. Verifies link mass / CoM against Leg_ver1 and every joint axis + sign.
"""
import adsk.core, adsk.fusion, math, traceback

MASTER_DOC = 'Leg_ver1'
MARKER = 'OP3_sim'          # stored in the root component description of the copy

LEG_ASM = {'l': 'MANUAL_OP3_L_LEG_ASM:1', 'r': 'MANUAL_OP3_R_LEG_ASM:1'}
# motor that drives each joint, in chain order (yaw, roll, pitch, knee, ankle pitch, ankle roll)
MOTORS = {
    'l': ['DUMMY_DC12_STD:1', 'DUMMY_DC12_STD:2', 'DUMMY_DC12_IDLE:1',
          'DUMMY_DC12_IDLE:2', 'DUMMY_DC12_IDLE:3', 'DUMMY_DC12_STD:3'],
    'r': ['DUMMY_DC12_STD:4', 'DUMMY_DC12_STD:5', 'DUMMY_DC12_IDLE:4',
          'DUMMY_DC12_IDLE:5', 'DUMMY_DC12_IDLE:6', 'DUMMY_DC12_STD:6'],
}
JOINTS = ['hip_yaw', 'hip_roll', 'hip_pitch', 'knee', 'ankle_pitch', 'ankle_roll']
LINKS = ['hip_yaw_link', 'hip_roll_link', 'thigh_link', 'shin_link', 'ankle_link', 'foot_link']
# (side, printed part, hidden metal original) -> zero-pose reference
ZERO_REFS = [
    ('l', 'P_H101_Clevis:1', 'FR12_H101_2:1'), ('l', 'P_Knee_Cap:1', 'FR12_X102-2:1'),
    ('l', 'P_HipAnkle_Block:2', 'FR12_X102_3:2'), ('l', 'P_H101_Clevis:2', 'FR12_H101_2:2'),
    ('l', 'P_Foot_Ribbed:1', 'FR12_E160_2:1'),
    ('r', 'P_H101_Clevis:5', 'FR12_H101_2:3'), ('r', 'P_Knee_Cap:2', 'FR12_X102-2:2'),
    ('r', 'P_HipAnkle_Block:4', 'FR12_X102_3:4'), ('r', 'P_H101_Clevis:4', 'FR12_H101_2:4'),
    ('r', 'P_Foot_Ribbed:2', 'FR12_E160_2:2'),
]


def _child(asm, name):
    for k in range(asm.childOccurrences.count):
        c = asm.childOccurrences.item(k)
        if c.name == name:
            return c
    raise RuntimeError('missing occurrence %s in %s' % (name, asm.name))


def _rot(m):
    a = m.asArray()
    return [[a[4 * i + j] for j in range(3)] for i in range(3)]


def build():
    app = adsk.core.Application.get()
    if app.activeDocument.name != MASTER_DOC:
        raise RuntimeError('Make %s the active document first (active: %s)'
                           % (MASTER_DOC, app.activeDocument.name))
    md = adsk.fusion.Design.cast(app.activeProduct)
    mroot = md.rootComponent
    legs = {s: mroot.occurrences.itemByName(n) for s, n in LEG_ASM.items()}

    # 1. zero pose
    for j in mroot.allAsBuiltJoints:
        j.jointMotion.rotationValue = 0
    adsk.doEvents()
    for s, p, o in ZERO_REFS:
        a = _child(legs[s], p).transform2.asArray()
        b = _child(legs[s], o).transform2.asArray()
        if max(abs(x - y) for x, y in zip(a, b)) > 1e-5:
            raise RuntimeError('Leg_ver1 is not in its zero pose at %s. Run the script again.' % p)

    # 2. link frames + bodies (copied in world coords, then moved into the link frame)
    frame = {'base_link': adsk.core.Matrix3D.create()}
    for s in 'lr':
        for i in range(6):
            frame[s + '_' + LINKS[i]] = _child(legs[s], MOTORS[s][i]).transform2.copy()
    tbm = adsk.fusion.TemporaryBRepManager.get()
    links, ref = [], {}
    for g in mroot.allRigidGroups:
        if g.name not in frame:
            raise RuntimeError('unexpected rigid group ' + g.name)
        inv = frame[g.name].copy()
        inv.invert()
        items = []
        m = cx = cy = cz = 0.0
        for occ in g.occurrences:
            if not occ.isLightBulbOn:
                continue
            for k in range(occ.bRepBodies.count):
                pb = occ.bRepBodies.item(k)
                if pb.isSolid and pb.isLightBulbOn:
                    pp = pb.physicalProperties
                    c = pp.centerOfMass
                    m += pp.mass
                    cx += pp.mass * c.x; cy += pp.mass * c.y; cz += pp.mass * c.z
                    t = tbm.copy(pb)
                    tbm.transform(t, inv)
                    items.append((t, pb.material, pb.appearance, occ.component.name))
        links.append((g.name, items))
        ref[g.name] = (m, cx / m, cy / m, cz / m)
    if len(links) != 13:
        raise RuntimeError('expected 13 rigid groups, found %d' % len(links))

    app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
    nd = adsk.fusion.Design.cast(app.activeProduct)
    nd.designType = adsk.fusion.DesignTypes.DirectDesignType
    nroot = nd.rootComponent
    nroot.description = MARKER
    mats, looks = {}, {}

    def mat(mm):
        if mm.name not in mats:
            mats[mm.name] = nd.materials.itemByName(mm.name) or nd.materials.addByCopy(mm, mm.name)
        return mats[mm.name]

    def look(a):
        if a is None:
            return None
        if a.name not in looks:
            looks[a.name] = nd.appearances.itemByName(a.name) or nd.appearances.addByCopy(a, a.name)
        return looks[a.name]

    for name, items in links:
        comp = nroot.occurrences.addNewComponent(frame[name]).component
        comp.name = name
        for t, mm, a, src in items:
            b = comp.bRepBodies.add(t)
            b.name = src
            b.material = mat(mm)
            lk = look(a)
            if lk:
                b.appearance = lk

    # 3. joints (as-built joints need parametric mode)
    nd.designType = adsk.fusion.DesignTypes.ParametricDesignType
    occs = {o.component.name: o for o in nroot.occurrences}
    for s in 'lr':
        chain = ['base_link'] + [s + '_' + l for l in LINKS]
        for i in range(6):
            child, parent = occs[chain[i + 1]], occs[chain[i]]
            geo = adsk.fusion.JointGeometry.createByPoint(
                child.component.originConstructionPoint.createForAssemblyContext(child))
            inp = nroot.asBuiltJoints.createInput(child, parent, geo)
            inp.setAsRevoluteJointMotion(adsk.fusion.JointDirections.ZAxisJointDirection)
            nroot.asBuiltJoints.add(inp).name = s + '_' + JOINTS[i]

    # 4. verify
    worst_c = worst_m = 0.0
    for name, (m, x, y, z) in ref.items():
        pp = occs[name].physicalProperties
        c = pp.centerOfMass
        worst_c = max(worst_c, math.sqrt((x - c.x) ** 2 + (y - c.y) ** 2 + (z - c.z) ** 2) * 10)
        worst_m = max(worst_m, abs(m - pp.mass) * 1000)
    bad = []
    js = {j.name: j for j in nroot.asBuiltJoints}
    for s in 'lr':
        for i in range(6):
            n = s + '_' + JOINTS[i]
            occ = occs[s + '_' + LINKS[i]]
            T0 = occ.transform2.copy()
            z0 = T0.getAsCoordinateSystem()[3]
            js[n].jointMotion.rotationValue = math.radians(10)
            adsk.doEvents()
            T1 = occ.transform2.copy()
            js[n].jointMotion.rotationValue = 0
            adsk.doEvents()
            R0, R1 = _rot(T0), _rot(T1)
            Rw = [[sum(R1[p][k] * R0[q][k] for k in range(3)) for q in range(3)] for p in range(3)]
            v = [Rw[2][1] - Rw[1][2], Rw[0][2] - Rw[2][0], Rw[1][0] - Rw[0][1]]
            nv = math.sqrt(sum(e * e for e in v))
            if nv < 1e-9 or (v[0] * z0.x + v[1] * z0.y + v[2] * z0.z) / nv < 0.999:
                bad.append(n)
    total = sum(r[0] for r in ref.values()) * 1000
    return ('OP3_sim rebuilt (unsaved). Save it as "OP3_sim".\n\n'
            'Links: 13, joints: 12, total mass: %.1f g\n'
            'vs Leg_ver1: worst CoM diff %.4f mm, worst mass diff %.3f g\n'
            'Joints on horn axis, + = CCW at horn: %d/12 %s'
            % (total, worst_c, worst_m, 12 - len(bad), bad if bad else ''))


def run(context):
    ui = adsk.core.Application.get().userInterface
    try:
        ui.messageBox(build(), 'build_sim_copy')
    except Exception:
        ui.messageBox('Failed:\n' + traceback.format_exc(), 'build_sim_copy')
