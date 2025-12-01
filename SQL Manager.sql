-- Project Decomposition Table, contains the project outline breakdown. 
create table public.project_decomposition (
  created_at timestamp with time zone not null default now(),
  client_id uuid not null default gen_random_uuid (),
  final_project_report jsonb null,
  team_mapper jsonb null,
  project_breakdown jsonb null,
  project_breakdown_critique jsonb null,
  updated_at timestamp with time zone null,
  project_id uuid not null default gen_random_uuid (),
  project_topic text null,
  overseer_decision text null,
  status public.project_decomp_status null,
  session_id character varying null,
  project_name text null,
  input text null,
  session_metadata jsonb null,
  select_freelancer_status public.project_decomp_status null default 'pending'::project_decomp_status,
  request_type public.request_type_project_decomp null,
  talent_based_output jsonb null,
  last_updated_at timestamp without time zone null default now(),
  project_budget real null,
  predicted_cost real null,
  duration_quantiles jsonb null,
  job_id uuid null,
  constraint project_decomposition_pkey primary key (project_id),
  constraint project_decomposition_project_id_key unique (project_id),
  constraint project_decomposition_session_id_key unique (session_id),
  constraint Project_Decomposition_client_id_fkey foreign KEY (client_id) references customers (id) on update CASCADE on delete RESTRICT
) TABLESPACE pg_default;

create index IF not exists idx_project_decomposition_job_id on public.project_decomposition using btree (job_id) TABLESPACE pg_default;

create trigger trigger_notify_orchestrator
after
update on project_decomposition for EACH row
execute FUNCTION notify_orchestrator_on_completion ();


-- Project Phases Table, contains the project phases and their detials. 
create table public.project_phases (
  id uuid not null default gen_random_uuid (),
  project_id uuid not null,
  phase_title text not null,
  phase_description text not null,
  tasks jsonb not null,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  phase_number smallint not null,
  phase_budget_pct real null,
  predicted_time_weeks real null,
  predicted_time_hours real null,
  predicted_time_months real null,
  task_times real[] null,
  duration_quantiles jsonb null,
  phase_embedding public.vector null,
  tasks_embedding public.vector null,
  approval_status text null default 'pending'::text,
  approved_by uuid null,
  approved_at timestamp with time zone null,
  approval_notes text null,
  constraint project_phases_pkey primary key (id),
  constraint uq_project_phases_project_title unique (project_id, phase_title),
  constraint project_phases_approved_by_fkey foreign KEY (approved_by) references auth.users (id),
  constraint project_phases_project_id_fkey foreign KEY (project_id) references project_decomposition (project_id) on delete CASCADE,
  constraint project_phases_approval_status_check check (
    (
      approval_status = any (
        array[
          'pending'::text,
          'awaiting_approval'::text,
          'approved'::text,
          'revision_requested'::text
        ]
      )
    )
  ),
  constraint project_phases_phase_number_check check ((phase_number > 0))
) TABLESPACE pg_default;

create index IF not exists idx_project_phases_project_id on public.project_phases using btree (project_id) TABLESPACE pg_default;

create index IF not exists idx_project_phases_title on public.project_phases using btree (phase_title) TABLESPACE pg_default;

create index IF not exists idx_project_phases_created_at on public.project_phases using btree (created_at) TABLESPACE pg_default;

create index IF not exists idx_project_phases_project_title on public.project_phases using btree (project_id, phase_title) TABLESPACE pg_default;


-- Project Phase Roles Table, contains the roles and their detials. 
create table public.project_phase_roles (
  id uuid not null default gen_random_uuid (),
  project_id uuid not null,
  phase_id uuid not null,
  role_id uuid not null,
  role_slot smallint not null,
  supplied_by text null,
  freelancer_id uuid null,
  can_span_phases boolean null default false,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone null default now(),
  role_skills text[] null,
  role_description jsonb null,
  specialized_role_name text null,
  role_budget_pct_phase real null,
  role_time_weeks real null,
  role_week_hours real null,
  skill_relevance_scores jsonb null,
  pre_selection_freelancer_ids uuid[] null,
  pre_selection_scores double precision[] null,
  pre_selection_k smallint null default '0'::smallint,
  pre_selection_updated_at timestamp with time zone null,
  recommended_freelancers uuid[] null,
  role_skill_embedding public.vector null,
  constraint project_phase_roles_pkey primary key (id),
  constraint project_phase_roles_freelancer_id_fkey foreign KEY (freelancer_id) references freelancers (id) on delete set null,
  constraint project_phase_roles_phase_id_fkey foreign KEY (phase_id) references project_phases (id) on delete CASCADE,
  constraint project_phase_roles_project_id_fkey foreign KEY (project_id) references project_decomposition (project_id) on delete CASCADE,
  constraint project_phase_roles_role_id_fkey foreign KEY (role_id) references roles (id) on update CASCADE on delete RESTRICT,
  constraint project_phase_roles_supplied_by_check check (
    (
      supplied_by = any (array['client'::text, 'platform'::text])
    )
  )
) TABLESPACE pg_default;

create unique INDEX IF not exists uq_phase_role_slot on public.project_phase_roles using btree (phase_id, role_id, role_slot) TABLESPACE pg_default;

create index IF not exists idx_project_phase_roles_project_role on public.project_phase_roles using btree (project_id, role_id) TABLESPACE pg_default;


create table public.roles (
  id uuid not null default extensions.uuid_generate_v4 (),
  name text not null,
  "Industry" text null,
  constraint roles_pkey primary key (id),
  constraint roles_name_key unique (name)
) TABLESPACE pg_default;