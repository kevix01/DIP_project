#!/usr/bin/env python3
"""
Script di Debug Visivo per la taratura delle soglie TTC sulle strisce pedonali.
Renderizza:
  1. Visuale Prospettica (con ROI strisce, impronte NERE per veicoli in ROI e ROSSE per violazioni)
  2. Visuale BEV (con ROI strisce, impronte NERE per veicoli in ROI e ROSSE per violazioni)
  3. Tabella dinamica TTC (Veicoli in ROI x Pedoni in ROI) con evidenziazione sotto-soglia.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter, deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bev_realtime as rt
import bev_realtime_v2 as bev_v2
from uod import style
from uod.textdraw import UILayer
from uod.viz import draw_cuboid, draw_dashed_polyline, stack_side_by_side
from violation_detector import ViolationDetector


class DebugV2Pipeline(bev_v2.V2Pipeline):
    """Estende V2Pipeline con colorazione diagnostica delle impronte e Tabella TTC."""

    def __init__(self, pose_model, aux_model=None, *, table_height: int = 300, **kw):
        super().__init__(pose_model, aux_model, **kw)
        self.table_height = int(table_height)
        self.current_debug_info: dict = {}
        self.roi_bev_points = None
        self.roi_image_points = None

    def _draw_panels(self, state):
        """Disegna i pannelli Camera e BEV con impronte NERE (in ROI) e ROSSE (Violazione)."""
        frame, tracks = state.frame, state.tracks
        aux_dets, aux_centers = state.aux_dets, state.aux_centers
        H, gi = state.H, state.gi
        src = frame.copy()

        dbg = self.current_debug_info or {}
        in_roi_veh_ids = dbg.get("vehicles_in_roi_ids", set())
        violator_veh_ids = dbg.get("violating_vehicle_ids", set())
        in_roi_ped_ids = {p.id for p in dbg.get("pedestrians_in_roi", [])}

        # --- 1. DISEGNO ROI IN PROSPETTIVA ---
        if self.roi_image_points is not None:
            roi_persp = self.roi_image_points
            overlay = src.copy()
            cv2.fillPoly(overlay, [roi_persp], color=(0, 255, 255))
            cv2.addWeighted(overlay, 0.18, src, 0.82, 0, src)
            cv2.polylines(src, [roi_persp], isClosed=True, color=(0, 255, 255),
                          thickness=2, lineType=cv2.LINE_AA)

        src_labels = []
        if self.draw_aux_box:
            for d in aux_dets:
                acol = style.THEME["aux"]
                x1, y1, x2, y2 = [float(v) for v in d.xyxy]
                draw_dashed_polyline(src, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                                     acol, thickness=2, dash=9, gap=6, halo=True)

        # --- 2. DISEGNO VEICOLI E PEDONI IN PROSPETTIVA ---
        for t in tracks:
            coasting = t.time_since_update > 0
            is_violator = (t.cls == 2 and t.id in violator_veh_ids)
            is_in_roi = (t.cls == 2 and t.id in in_roi_veh_ids)

            # Colori in RGB per lo stile UI/cuboidi e BGR per OpenCV diretto
            if is_violator:
                col_rgb = (255, 30, 30)     # ROSSO per chi commette violazione
                fp_bgr = (0, 0, 255)
                fp_border_bgr = (255, 255, 255)
                fill_alpha = 0.55
            elif is_in_roi:
                col_rgb = (20, 20, 20)      # NERO per i veicoli dentro la ROI
                fp_bgr = (0, 0, 0)
                fp_border_bgr = (0, 255, 255)
                fill_alpha = 0.68
            else:
                col_rgb = style.instance_color(t.cls, t.id, source="pose")
                fp_bgr = None
                fp_border_bgr = None
                fill_alpha = 0.0

            # Se il veicolo è nella ROI o in violazione, colora esplicitamente l'impronta a terra in prospettiva
            if fp_bgr is not None:
                g_pts = None
                if getattr(t, "ground", None) is not None and t.ground.shape == (4, 2):
                    g_pts = t.ground
                elif t.kpts.shape[0] >= 8 and gi:
                    g_pts = t.kpts[list(gi)]
                if g_pts is not None and np.all(np.isfinite(g_pts)):
                    poly_2d = np.round(g_pts).astype(np.int32)
                    ov_fp = src.copy()
                    cv2.fillPoly(ov_fp, [poly_2d], fp_bgr, lineType=cv2.LINE_AA)
                    cv2.addWeighted(ov_fp, fill_alpha, src, 1.0 - fill_alpha, 0, src)
                    cv2.polylines(src, [poly_2d], isClosed=True, color=fp_border_bgr,
                                  thickness=2, lineType=cv2.LINE_AA)

            # Disegna cuboide 3D
            if self.show_3d and t.kpts.shape[0] >= 8:
                draw_cuboid(src, t.kpts, gi, col_rgb, dim=coasting, fill=False)

            x1, y1 = t.box_xyxy[0], t.box_xyxy[1]
            bh = float(t.box_xyxy[3] - t.box_xyxy[1])
            src_labels.append((x1, y1, t.cls, t.id, coasting, bh, is_in_roi, is_violator, t.id in in_roi_ped_ids))

        # --- 3. DISEGNO PANNELLO BEV ---
        bev = self.viewport.blank_canvas(radar=True)
        if self.use_homography and self.show_fov:
            self.viewport.draw_fov(bev, H, state.img_wh)

        # Disegna ROI sulla BEV
        if self.roi_bev_points is not None:
            roi_scaled = self.roi_bev_points * state.bev_unit
            roi_canvas = self.viewport.rect_to_canvas(roi_scaled)
            if len(roi_canvas) > 0 and np.all(np.isfinite(roi_canvas)):
                roi_canvas_pts = np.round(roi_canvas).astype(np.int32)
                overlay_bev = bev.copy()
                cv2.fillPoly(overlay_bev, [roi_canvas_pts], color=(0, 255, 255))
                cv2.addWeighted(overlay_bev, 0.18, bev, 0.82, 0, bev)
                cv2.polylines(bev, [roi_canvas_pts], isClosed=True, color=(0, 255, 255),
                              thickness=2, lineType=cv2.LINE_AA)

        bev_labels = []
        present = Counter()
        if self.use_homography:
            fps = []
            for t in tracks:
                if t.bev_quad is None:
                    continue
                poly = self.viewport.rect_to_canvas(t.bev_quad)
                if len(poly) != 4 or not np.all(np.isfinite(poly)):
                    continue
                fps.append((t, poly))
                c = poly.mean(axis=0)
                bev_labels.append((c[0], c[1], t.id, t.cls))

            if self.trails:
                cur = set()
                for (cx, cy, tid, _) in bev_labels:
                    cur.add(tid)
                    self._trails.setdefault(tid, deque(maxlen=self.trail_len)).append((cx, cy))
                for tid in [k for k in self._trails if k not in cur]:
                    del self._trails[tid]
                self._draw_trails(bev, {t.id: t.cls for t, _ in fps})

            track_polys = []
            for (t, poly) in fps:
                is_violator = (t.cls == 2 and t.id in violator_veh_ids)
                is_in_roi = (t.cls == 2 and t.id in in_roi_veh_ids)

                if is_violator:
                    # Impronta ROSSA in BEV per i veicoli in violazione
                    self.viewport.draw_footprint_poly(
                        bev, poly, (255, 0, 0), fill_alpha=0.80, thickness=3)
                    pts_i = np.round(poly).astype(np.int32)
                    cv2.polylines(bev, [pts_i], True, (0, 0, 255), 2, cv2.LINE_AA)
                elif is_in_roi:
                    # Impronta NERA in BEV per i veicoli nella ROI (con bordo chiaro per contrasto)
                    pts_i = np.round(poly).astype(np.int32)
                    cv2.fillPoly(bev, [pts_i], (0, 0, 0), lineType=cv2.LINE_AA)
                    cv2.polylines(bev, [pts_i], True, (220, 220, 220), 2, cv2.LINE_AA)
                else:
                    col = style.instance_color(t.cls, t.id, source="pose")
                    alpha = 0.30 if t.time_since_update == 0 else 0.12
                    self.viewport.draw_footprint_poly(
                        bev, poly, col, fill_alpha=alpha,
                        thickness=2 if t.time_since_update == 0 else 1)
                track_polys.append(poly)

            self._draw_aux_bev(bev, aux_dets, aux_centers, H, track_polys)

        for t in tracks:
            present[style.class_category(t.cls, "pose")] += 1
        self.viewport.draw_camera(bev)
        return src, bev, present, src_labels, bev_labels

    def _render(self, state):
        """Compone Camera | BEV sopra e la Tabella di Debug TTC sotto."""
        src, bev, present, src_labels, bev_labels = self._draw_panels(state)
        composite = stack_side_by_side(src, bev, gap=2)
        bev_x = src.shape[1] + 2

        ui = UILayer()
        # Intestazione ed etichette sui pannelli Camera e BEV
        self._draw_debug_overlays(ui, composite, state, present, src_labels, bev_labels, bev_x)
        top_rendered = ui.render(composite)

        # Costruisce la Tabella TTC in basso a larghezza piena
        table_img = self._render_ttc_table(top_rendered.shape[1], state)
        return np.vstack([top_rendered, table_img])

    def _draw_debug_overlays(self, ui, composite, state, present, src_labels, bev_labels, bev_x):
        """Disegna gli ID di tutti i veicoli/pedoni rilevanti sia su Camera che su BEV."""
        T = style.THEME
        Wc, Hc = composite.shape[1], composite.shape[0]
        head = 42
        ui.panel((0, 0), (Wc, head), (0, 0, 0), alpha=0.88, radius=0)
        ui.line((0, head), (Wc, head), T["separator"], width=1, alpha=1.0)
        ui.text(20, head / 2, "DEBUG VIOLAZIONI & TTC - UrbanOmniDetect", size=19,
                font="bold", color=T["panel_text"], anchor="lm", shadow=False)
        ui.text(Wc - 20, head / 2, f"Frame #{state.frame_idx}   {self._fps_text(state)}",
                size=15, font="mono", color=T["muted_text"], anchor="rm", shadow=False)

        # Etichette sulla vista prospettica (mostra sempre i veicoli/pedoni in ROI o violazione)
        for (x, y, cls, tid, coasting, bh, is_in_roi, is_violator, ped_in_roi) in src_labels:
            prefix = "P" if cls == 0 else "V"
            if is_violator:
                txt = f"{prefix}{tid} [VIOLAZIONE]"
                bg_col = (200, 20, 20)
                dot_col = (255, 255, 255)
            elif is_in_roi:
                txt = f"{prefix}{tid} [IN ROI]"
                bg_col = (10, 10, 10)
                dot_col = (255, 255, 0)
            elif ped_in_roi:
                txt = f"{prefix}{tid} [STRISCE]"
                bg_col = (20, 90, 40)
                dot_col = (240, 178, 74)
            else:
                if bh < self.label_min_h:
                    continue
                txt = f"{prefix}{tid}"
                bg_col = (0, 0, 0)
                dot_col = style.instance_color(cls, tid, source="pose")

            bev_v2._pill(ui, float(x) + 1, max(float(y), head + 18) - 6, txt,
                         size=14, dot=dot_col, fg=(248, 248, 250), bg=bg_col,
                         alpha=0.88, pad=(8, 4), anchor="ld")

        # Etichette ID sulla BEV
        for (cx, cy, tid, cls) in bev_labels:
            prefix = "P" if cls == 0 else "V"
            ui.chip(bev_x + cx, cy, f"{prefix}{tid}", size=14,
                    fg=(255, 255, 255), bg=(15, 15, 18), alpha=0.88)

    def _render_ttc_table(self, width: int, state) -> np.ndarray:
        """Disegna il pannello tabellare con i TTC tra ogni Veicolo nella ROI e ogni Pedone nella ROI."""
        H_tab = self.table_height
        panel = np.full((H_tab, width, 3), (18, 18, 22), dtype=np.uint8)

        dbg = self.current_debug_info or {}
        vehs = dbg.get("vehicles_in_roi", [])
        peds = dbg.get("pedestrians_in_roi", [])
        ttc_mat = dbg.get("ttc_matrix", {})
        min_ttc_map = dbg.get("min_ttc_per_veh", {})
        violators = dbg.get("violating_vehicle_ids", set())
        newly_trig = dbg.get("newly_triggered_ids", set())
        thresh = float(dbg.get("ttc_threshold", 2.5))

        # Linea di separazione superiore gialla/grigia
        cv2.line(panel, (0, 0), (width, 0), (0, 210, 255), 2)

        # Barra titolo della tabella
        header_h = 38
        cv2.rectangle(panel, (0, 2), (width, header_h), (28, 28, 35), -1)
        summary = (f"TABELLA TTC (Veicoli in ROI: {len(vehs)} | Pedoni in ROI: {len(peds)} | "
                   f"Soglia TTC: {thresh:.2f} s | Violazioni Totali Video: {len(violators)})")
        cv2.putText(panel, summary, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (240, 240, 245), 2, cv2.LINE_AA)

        # Legenda rapida a destra
        legend_items = [
            ((0, 0, 0), (220, 220, 220), "Veicolo in ROI (Nero)"),
            ((0, 0, 220), (255, 255, 255), f"Violazione / TTC <= {thresh:.1f}s (Rosso)"),
            ((35, 120, 45), (255, 255, 255), "TTC > Soglia (Verde)"),
            ((55, 55, 62), (200, 200, 205), "Divergente TTC < 0 (Grigio)"),
        ]
        lx = width - 16
        for bg_c, border_c, l_txt in reversed(legend_items):
            (tw, th), _ = cv2.getTextSize(l_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            box_w = tw + 28
            lx -= box_w
            cv2.rectangle(panel, (lx, 8), (lx + 16, 28), bg_c, -1)
            cv2.rectangle(panel, (lx, 8), (lx + 16, 28), border_c, 1)
            cv2.putText(panel, l_txt, (lx + 22, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (225, 225, 230), 1, cv2.LINE_AA)
            lx -= 14

        # Se non ci sono veicoli o pedoni nella ROI, mostra un messaggio diagnostico chiaro
        if not vehs or not peds:
            msg_y = header_h + (H_tab - header_h) // 2
            if not vehs and not peds:
                msg = "Nessun veicolo e nessun pedone attualmente all'interno della ROI delle strisce."
            elif not vehs:
                ped_list = ", ".join(f"P{p.id}" for p in peds)
                msg = f"Pedoni sulle strisce: [{ped_list}] — Nessun veicolo ha l'impronta sulla ROI."
            else:
                veh_list = ", ".join(f"V{v.id}" for v in vehs)
                msg = f"Veicoli sulla ROI: [{veh_list}] (Impronta Nera/Rossa) — Nessun pedone dentro la ROI."
            cv2.putText(panel, msg, (24, msg_y), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                        (175, 175, 185), 2, cv2.LINE_AA)
            return panel

        # Configurazione griglia tabellare
        top_y = header_h + 6
        col_head_h = 44
        max_rows = 6
        vehs_shown = vehs[:max_rows]
        avail_h = H_tab - top_y - col_head_h - 8
        row_h = max(28, min(42, avail_h // max(len(vehs_shown), 1)))

        # Colonne fisse a sinistra: Veicolo | Stato | Pos BEV | Vel BEV | Min TTC
        fixed_cols = [
            ("VEICOLO", 105),
            ("STATO", 145),
            ("POS BEV (x,y)", 150),
            ("VEL BEV |v|", 145),
            ("MIN TTC", 125),
        ]
        fixed_w = sum(w for _, w in fixed_cols) + 16
        rem_w = max(200, width - fixed_w - 16)
        max_ped_cols = min(len(peds), max(1, rem_w // 135))
        peds_shown = peds[:max_ped_cols]
        ped_col_w = rem_w // max(len(peds_shown), 1)

        # Disegna intestazioni colonne fisse
        cx = 16
        cv2.rectangle(panel, (16, top_y), (width - 16, top_y + col_head_h), (36, 36, 46), -1)
        for cname, cw in fixed_cols:
            cv2.putText(panel, cname, (cx + 8, top_y + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (200, 205, 215), 1, cv2.LINE_AA)
            cv2.line(panel, (cx + cw, top_y), (cx + cw, top_y + col_head_h + len(vehs_shown) * row_h),
                     (55, 55, 65), 1)
            cx += cw

        # Disegna intestazioni colonne Pedoni in ROI
        for ped in peds_shown:
            p_speed = float(np.linalg.norm(ped.bev_vel))
            title_p = f"PEDONE P{ped.id}"
            sub_p = f"({ped.bev_pos[0]:+.1f},{ped.bev_pos[1]:+.1f}) v={p_speed:.1f}"
            cv2.putText(panel, title_p, (cx + 8, top_y + 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (74, 178, 240), 2, cv2.LINE_AA)
            cv2.putText(panel, sub_p, (cx + 8, top_y + 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (170, 175, 185), 1, cv2.LINE_AA)
            cv2.line(panel, (cx + ped_col_w, top_y),
                     (cx + ped_col_w, top_y + col_head_h + len(vehs_shown) * row_h),
                     (55, 55, 65), 1)
            cx += ped_col_w

        # Disegna una riga per ogni Veicolo nella ROI
        for r_idx, veh in enumerate(vehs_shown):
            ry = top_y + col_head_h + r_idx * row_h
            is_viol = veh.id in violators
            is_new = veh.id in newly_trig

            row_bg = (32, 22, 25) if is_viol else ((24, 24, 30) if r_idx % 2 == 0 else (28, 28, 35))
            cv2.rectangle(panel, (16, ry), (width - 16, ry + row_h), row_bg, -1)
            cv2.line(panel, (16, ry + row_h), (width - 16, ry + row_h), (50, 50, 60), 1)

            v_speed = float(np.linalg.norm(veh.bev_vel))
            min_ttc = min_ttc_map.get(veh.id, float('inf'))
            min_ttc_str = f"{min_ttc:.2f} s" if np.isfinite(min_ttc) else "N/A"

            if is_new:
                st_txt = "INFRAZIONE ORA!"
                st_bg = (0, 0, 235)
                st_fg = (255, 255, 255)
            elif is_viol:
                st_txt = "VIOLATORE (ROSSO)"
                st_bg = (0, 0, 165)
                st_fg = (255, 255, 255)
            else:
                st_txt = "IN ROI (NERO)"
                st_bg = (0, 0, 0)
                st_fg = (0, 255, 255)

            cx = 16
            # 1. ID Veicolo
            cw = fixed_cols[0][1]
            cv2.putText(panel, f"V{veh.id}", (cx + 10, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            cx += cw

            # 2. Badge Stato
            cw = fixed_cols[1][1]
            cv2.rectangle(panel, (cx + 4, ry + 4), (cx + cw - 4, ry + row_h - 4), st_bg, -1)
            cv2.rectangle(panel, (cx + 4, ry + 4), (cx + cw - 4, ry + row_h - 4), (180, 180, 190), 1)
            cv2.putText(panel, st_txt, (cx + 8, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, st_fg, 1, cv2.LINE_AA)
            cx += cw

            # 3. Posizione BEV
            cw = fixed_cols[2][1]
            pos_str = f"({veh.bev_pos[0]:+.2f}, {veh.bev_pos[1]:+.2f})"
            cv2.putText(panel, pos_str, (cx + 8, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 225), 1, cv2.LINE_AA)
            cx += cw

            # 4. Velocità BEV
            cw = fixed_cols[3][1]
            vel_str = f"{v_speed:.2f} BEU/s"
            cv2.putText(panel, vel_str, (cx + 8, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 225), 1, cv2.LINE_AA)
            cx += cw

            # 5. Min TTC
            cw = fixed_cols[4][1]
            min_col = (60, 60, 255) if min_ttc <= thresh else (120, 230, 120)
            cv2.putText(panel, min_ttc_str, (cx + 10, ry + row_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, min_col, 2, cv2.LINE_AA)
            cx += cw

            # 6. Celle TTC verso ogni Pedone
            for ped in peds_shown:
                ttc_val = ttc_mat.get((veh.id, ped.id), float('inf'))
                c_x1, c_y1 = cx + 3, ry + 3
                c_x2, c_y2 = cx + ped_col_w - 3, ry + row_h - 3

                if 0.0 <= ttc_val <= thresh:
                    # SOTTO SOGLIA -> Evidenziazione ROSSA
                    cell_bg = (20, 20, 220)
                    cell_border = (0, 255, 255)
                    cell_txt = f"{ttc_val:.2f}s [SOGLIA!]"
                    txt_col = (255, 255, 255)
                    font_w = 2
                elif np.isfinite(ttc_val) and ttc_val > thresh:
                    if ttc_val <= thresh * 1.5:
                        # Vicino alla soglia -> Arancio/Giallo
                        cell_bg = (20, 110, 175)
                        cell_border = (60, 180, 240)
                    else:
                        # Ampiamente sopra soglia -> Verde
                        cell_bg = (30, 95, 40)
                        cell_border = (60, 160, 75)
                    cell_txt = f"{ttc_val:.2f} s"
                    txt_col = (245, 245, 245)
                    font_w = 1
                elif ttc_val == -2.0:
                    # Pedone già passato / in allontanamento dalla corsia del veicolo
                    cell_bg = (45, 45, 52)
                    cell_border = (75, 75, 85)
                    cell_txt = "PASSATO (oltre)"
                    txt_col = (175, 175, 185)
                    font_w = 1
                elif ttc_val == -3.0:
                    # d_CPA > conflict_radius (nessuna rotta di collisione)
                    cell_bg = (45, 45, 52)
                    cell_border = (75, 75, 85)
                    cell_txt = "NO COLL (corsia)"
                    txt_col = (175, 175, 185)
                    font_w = 1
                elif ttc_val < 0.0:
                    # Traiettorie divergenti
                    cell_bg = (45, 45, 52)
                    cell_border = (75, 75, 85)
                    cell_txt = "DIV (allont.)"
                    txt_col = (175, 175, 185)
                    font_w = 1
                else:
                    # Velocità relativa ~ 0 -> Infinito
                    cell_bg = (45, 45, 52)
                    cell_border = (75, 75, 85)
                    cell_txt = "INF (fermi)"
                    txt_col = (175, 175, 185)
                    font_w = 1

                cv2.rectangle(panel, (c_x1, c_y1), (c_x2, c_y2), cell_bg, -1)
                cv2.rectangle(panel, (c_x1, c_y1), (c_x2, c_y2), cell_border, 1)
                cv2.putText(panel, cell_txt, (cx + 10, ry + row_h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, font_w, cv2.LINE_AA)
                cx += ped_col_w

        return panel


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Aggiunge parametri specifici per il debug
    dbg_parser = argparse.ArgumentParser(add_help=False)
    dbg_parser.add_argument("--table-height", type=int, default=280,
                            help="Altezza in pixel della tabella TTC in basso (default: 280)")
    dbg_parser.add_argument("--no-save-violations", action="store_true",
                            help="Disabilita il salvataggio su disco di JSON/JPG durante il debug")
    dbg_args, rest_argv = dbg_parser.parse_known_args(argv)

    v2args, rest = bev_v2.build_v2_argparser().parse_known_args(rest_argv)
    if not any(a.startswith("--aux-model") for a in rest):
        rest += ["--aux-model", "none"]
    args = rt.build_argparser().parse_args(rest)

    src = bev_v2.FrameSource(args.input, source_type=getattr(v2args, "source_type", "auto"))
    fps = float(v2args.refine_fps) or src.fps
    src_w = int(src.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0

    canvas = None
    if v2args.bev_canvas:
        w, h = v2args.bev_canvas.lower().split("x")
        canvas = (int(w), int(h))
    elif v2args.layout == "dashboard":
        canvas = (max(360, bev_v2.FINAL_W - src_w) if src_w else 640, bev_v2.FINAL_H)

    bev_v2.V2Viewport.DEFAULT_OBJECT_PX = float(v2args.object_px)
    smooth_win = int(args.smooth) if (args.smooth and args.smooth >= 3) else 11

    # Assegna a bev_v2.V2Pipeline.CONFIG perché V2Pipeline.__init__ legge da lì
    bev_v2.V2Pipeline.CONFIG = dict(
        total_frames=src.total_frames,
        kp_vis=v2args.kp_vis,
        suppress_nested=v2args.suppress_nested,
        class_conf=bev_v2._parse_class_conf(v2args.class_conf),
        bev_canvas=canvas,
        layout="side",
        caption=v2args.caption,
        show_coast_mark=v2args.coast_mark,
        label_min_h=v2args.label_min_h,
        src_fps=fps,
        smooth_win=smooth_win,
        title=(0.0, 0.0, 0.0, 0.0),
        track_high_conf=v2args.track_high_conf,
        track_ocm_weight=v2args.track_ocm_weight,
        bev_unit=v2args.bev_unit,
        show_bev_ids=True,
    )

    gi = ([int(x) for x in args.ground_indices.split(",")]
          if args.ground_indices else None)
    pose = rt.load_detector(args.kp_model, device=args.device, imgsz=args.kp_imgsz,
                            export=args.export, half=args.half, task="pose")

    pipe = DebugV2Pipeline(
        pose, None,
        table_height=dbg_args.table_height,
        kp_imgsz=args.kp_imgsz,
        kp_conf=args.kp_conf,
        device=args.device,
        ground_indices=gi,
        use_homography=(args.homography != "none"),
        show_3d=not args.no_3d,
        snap_rect=not args.no_snap_rect,
        trails=not args.no_trails,
        trail_len=args.trail_len,
        stabilize=not args.no_stabilize,
        bev_window=args.bev_window,
        show_fov=not args.no_fov,
        tracker_kwargs=dict(max_age=args.max_age, min_hits=args.min_hits),
    )

    violation_detector = ViolationDetector(
        video_name=args.input,
        roi_file=v2args.roi_file,
        bev_unit=pipe._bev_unit_locked,
        ttc_threshold=v2args.ttc_thresh,
        timeout_frames=v2args.timeout_frames,
        save_files=not dbg_args.no_save_violations,
    )
    pipe.roi_bev_points = violation_detector.roi.bev_points
    pipe.roi_image_points = violation_detector.roi.image_points

    sink = bev_v2._FFSink(args, fps)
    n_out = 0
    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            if frame is None:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            raw_state = pipe.step(frame)
            ready_state = pipe.lag_buffer.push(raw_state, pipe._tracker_map)

            if ready_state is not None:
                pipe.current_debug_info = violation_detector.process_frame(
                    frame=ready_state.frame.copy(),
                    tracks=ready_state.tracks,
                    frame_idx=ready_state.frame_idx,
                    H_inv=pipe.static_H_inv,
                )
                out_img = pipe.render(ready_state)
                dt = time.perf_counter() - t0
                pipe.fps_core = pipe.fps_render = rt._ema(pipe.fps_core, dt)
                n_out += 1
                if not sink.emit(out_img) or (args.max_frames and n_out >= args.max_frames):
                    break

        if not src.is_stream and not (args.max_frames and n_out >= args.max_frames):
            for tail_state in pipe.lag_buffer.flush_remaining(pipe._tracker_map):
                if tail_state is not None:
                    pipe.current_debug_info = violation_detector.process_frame(
                        frame=tail_state.frame.copy(),
                        tracks=tail_state.tracks,
                        frame_idx=tail_state.frame_idx,
                        H_inv=pipe.static_H_inv,
                    )
                    out_img = pipe.render(tail_state)
                    n_out += 1
                    if not sink.emit(out_img):
                        break
    finally:
        src.release()
        sink.close()
        if args.display:
            cv2.destroyAllWindows()

    print(f"\nCompletato Debug: {n_out} frame renderizzati ({pipe.fps_core:.1f} fps)")
    return n_out


if __name__ == "__main__":
    main()