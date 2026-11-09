// Added by HexpyGames for Butler (2026), under the Apache License 2.0; see NOTICE.
//! Damage redraw: an animation tick that re-encodes only the work intersecting the animated
//! primitives, over the complete frame the previous full encode left in the offscreen scene target.
use super::*;

/// Whether compositing `filter` reads a blurred backdrop that the cache can supply.
pub(super) fn uses_blurred_backdrop(filter: &BackdropFilter) -> bool {
    BlurKernel::for_radius(filter.max_blur_radius()).is_some() || filter.lens.is_active()
}

impl DirectXRenderer {
    /// Presents an animation tick of the scene last passed to `draw`: a damage redraw over the
    /// retained frame when the scene allows it, a full encode otherwise.
    pub(crate) fn draw_animation_frame(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        if self.skip_draws {
            return Ok(());
        }
        self.animation_time_ms = gpui::animation_time_ms();
        if !self.render_damage(scene)? {
            self.render(scene, background_appearance, false)?;
        }
        self.present()
    }

    /// Encodes the scene's damage plan over the retained frame and blits it to the swapchain;
    /// `false` when this frame needs a full encode (no retained frame for this scene, a `Full`
    /// plan, or a missing cache entry).
    pub(super) fn render_damage(&mut self, scene: &Scene) -> Result<bool> {
        let Some(DamagePlan::Partial(plan)) = scene.damage_plan_over(self.retained) else {
            return Ok(false);
        };
        let generation = scene.generation();
        if self.retained_atlas_epoch != self.atlas.removal_epoch() {
            return Ok(false);
        }
        let Some(scissor) = self.clamped_scissor(plan.rect) else {
            return Ok(false);
        };

        let target = size(
            DevicePixels(self.width as i32),
            DevicePixels(self.height as i32),
        );
        if !gpui_render::damage_fits(scissor, target) {
            return Ok(false);
        }
        if self.frame_keys.0 != generation {
            self.frame_keys = (
                generation,
                gpui_render::backdrop_key::backdrop_keys(scene, target),
            );
        }
        if !self.backdrops_cached(scene, plan) {
            return Ok(false);
        }
        if self.uploaded_generation != Some(generation) {
            self.upload_scene_buffers(scene)?;
        }
        self.upload_globals()?;

        // The fill needs a scratch target.
        let mut requirements = plan.requirements;
        requirements.uses_offscreen_target = true;
        requirements.uses_blur_target = true;
        let device = &self.devices.as_ref().context("devices missing")?.device;
        let resources = self.resources.as_mut().context("resources missing")?;
        if requirements.uses_path_target {
            resources.ensure_path_resources(device)?;
        }
        resources.ensure_blur_resources(device, &requirements)?;

        let blur = resources.blur.as_ref().context("blur resources missing")?;
        let scene_rtv = blur.scene.target.clone();
        let scene_srv = blur.scene.source.clone();
        let scratch_source = blur
            .scratch
            .as_ref()
            .and_then(|[ping, _]| ping.source.clone());
        let groups = blur
            .groups
            .iter()
            .map(|group| (group.target.clone(), group.source.clone()))
            .collect::<SmallVec<[_; MAX_FILTER_GROUP_DEPTH]>>();
        let swapchain_rtv = resources.render_target_view.clone();
        let viewport = resources.viewport;
        let ctx = self
            .devices
            .as_ref()
            .context("devices missing")?
            .device_context
            .clone();

        unsafe {
            ctx.OMSetRenderTargets(Some(slice::from_ref(&scene_rtv)), None);
            ctx.RSSetViewports(Some(slice::from_ref(&viewport)));
        }
        self.active_render_target = scene_rtv.clone();

        // The offscreen frame clears to transparent black, so refilling the damage rectangle with
        // it gives the redraw the same starting point.
        self.dx_scissored(&scissor, || {
            self.dx_blur_pass(
                &self.pipelines.blur_downsample_vertex,
                &self.pipelines.blur_downsample_fragment,
                &self.pipelines.blur_blend_replace,
                &scene_rtv,
                &scratch_source,
                BlurUniforms::fill([0.0; 4]),
                &viewport,
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                3,
                false,
            )
        })?;

        self.damage_scissor = Some(windows::Win32::Foundation::RECT {
            left: scissor.x as i32,
            top: scissor.y as i32,
            right: (scissor.x + scissor.width) as i32,
            bottom: (scissor.y + scissor.height) as i32,
        });
        self.restore_rasterizer();
        let mut targets = TargetStack::new((scene_rtv, scene_srv.clone()));
        let result = self.encode_commands(scene, &plan.commands, &mut targets, &groups, Some(plan));
        self.damage_scissor = None;
        self.restore_rasterizer();
        self.active_render_target = None;
        if let Err(error) = result {
            self.retained = None;
            return Err(error);
        }
        self.dx_blit(&scene_srv, &swapchain_rtv)?;
        self.retained = Some(generation);
        Ok(true)
    }

    /// Whether every kept backdrop filter the plan does not re-blur can restore its blur from the cache.
    fn backdrops_cached(&self, scene: &Scene, plan: &DamageRedraw) -> bool {
        let Some(blur) = self
            .resources
            .as_ref()
            .and_then(|resources| resources.blur.as_ref())
        else {
            return false;
        };
        let cache = blur.cache.borrow();
        plan.commands.iter().all(|command| {
            let RenderCommand::Batch(PrimitiveBatch::BackdropFilters(range)) = command else {
                return true;
            };
            range.clone().all(|index| {
                let filter = &scene.backdrop_filters[index];
                filter.wallpaper.is_some()
                    || plan.reblur.contains(&index)
                    || !uses_blurred_backdrop(filter)
                    || self
                        .frame_keys
                        .1
                        .get(index)
                        .copied()
                        .flatten()
                        .is_some_and(|key| cache.contains(key))
            })
        })
    }

    /// The damage rectangle clamped to the target; `None` when nothing of it lies on the target.
    fn clamped_scissor(&self, rect: Bounds<ScaledPixels>) -> Option<ScissorRectangle> {
        let clamp = |value: f32, limit: u32| value.clamp(0.0, limit as f32);
        let x0 = clamp(rect.origin.x.0.floor(), self.width);
        let y0 = clamp(rect.origin.y.0.floor(), self.height);
        let x1 = clamp((rect.origin.x.0 + rect.size.width.0).ceil(), self.width);
        let y1 = clamp((rect.origin.y.0 + rect.size.height.0).ceil(), self.height);
        let scissor = ScissorRectangle {
            x: x0 as u32,
            y: y0 as u32,
            width: (x1 - x0).max(0.0) as u32,
            height: (y1 - y0).max(0.0) as u32,
        };
        (!scissor.is_empty()).then_some(scissor)
    }
}
