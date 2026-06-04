alter table public.scenarios enable row level security;

drop policy if exists "Users can view own scenarios" on public.scenarios;
create policy "Users can view own scenarios"
on public.scenarios
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists "Users can insert own scenarios" on public.scenarios;
create policy "Users can insert own scenarios"
on public.scenarios
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists "Users can delete own scenarios" on public.scenarios;
create policy "Users can delete own scenarios"
on public.scenarios
for delete
to authenticated
using (auth.uid() = user_id);
