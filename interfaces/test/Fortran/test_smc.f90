! test_smc.f90 - tests for the Fortran binding of libsmc (interfaces/include/smc.f90).
!
! The Fortran counterpart of interfaces/test/C/test_api.c and test_coloring.c:
! the same contract through the Fortran binding, so that a drift between smc.h
! and smc.f90 is caught here rather than as silent memory corruption.
!
! Four Fortran-specific rules this file follows:
!   1. anything whose address is taken with c_loc carries the target attribute;
!   2. a buffer length may legitimately be 0, and integer(c_size_t) is signed in
!      Fortran, so 0 - 1 reaches C as SIZE_MAX (a valid, enormous promise):
!      never understate a length without first checking that it is positive;
!   3. a result is bidirectional when Br_cols > 0, never when Br_len > 0, which
!      can be 0 for a bidirectional result whose row coloring is empty;
!   4. trim() is never applied to a character array element -- that form kills
!      the process under MinGW.  Copy into a scalar and use X(1:len_trim(X)).
!
! Compile (after building libsmc with juliac - see interfaces/README.md):
!   gfortran -O2 -o interfaces/build/test_smc_fortran \
!       interfaces/test/Fortran/test_smc.f90 \
!       -I interfaces/include \
!       interfaces/build/lib/libsmc.so
!
! Exit code: 0 if all checks pass, 1 otherwise.

program test_smc
  use iso_c_binding
  use iso_fortran_env, only: output_unit
  implicit none
  include 'smc.f90'          ! must come after implicit none

  ! Harness state and problem data, visible to the contained procedures by host
  ! association.

  integer, parameter :: MAXDIM = 12
  integer, parameter :: MAXNNZ = 64

  ! A sparsity pattern with values, in CSC form.  The arrays are oversized so
  ! that one fixed-size type holds every test matrix; only the first n+1 and nnz
  ! entries are ever passed on.
  type :: TestMatrix
    character(len=24) :: name
    integer(c_int)    :: m, n, nnz
    integer(c_int)    :: colptr(MAXDIM+1)
    integer(c_int)    :: rowval(MAXNNZ)
    real(c_double)    :: nzval(MAXNNZ)
  end type TestMatrix

  ! target, because c_loc is taken of their colptr / rowval components.
  type(TestMatrix), target :: NONSYM(2), SYMM(2)      ! index_base 0
  type(TestMatrix), target :: NONSYM1(2), SYMM1(2)    ! the same, index_base 1

  ! The six supported triples, one per column (Fortran is column-major).
  integer(c_int), parameter :: SUPPORTED(3,6) = reshape( [                     &
      SMC_NONSYMMETRIC, SMC_COLUMN,        SMC_DIRECT,                         &
      SMC_NONSYMMETRIC, SMC_ROW,           SMC_DIRECT,                         &
      SMC_SYMMETRIC,    SMC_COLUMN,        SMC_DIRECT,                         &
      SMC_SYMMETRIC,    SMC_COLUMN,        SMC_SUBSTITUTION,                   &
      SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_DIRECT,                         &
      SMC_NONSYMMETRIC, SMC_BIDIRECTIONAL, SMC_SUBSTITUTION ], [3,6] )

  ! The six triples that are not a SparseMatrixColorings problem; see smc.f90.
  integer(c_int), parameter :: UNSUPPORTED(3,6) = reshape( [                   &
      SMC_NONSYMMETRIC, SMC_COLUMN,        SMC_SUBSTITUTION,                   &
      SMC_NONSYMMETRIC, SMC_ROW,           SMC_SUBSTITUTION,                   &
      SMC_SYMMETRIC,    SMC_ROW,           SMC_DIRECT,                         &
      SMC_SYMMETRIC,    SMC_ROW,           SMC_SUBSTITUTION,                   &
      SMC_SYMMETRIC,    SMC_BIDIRECTIONAL, SMC_DIRECT,                         &
      SMC_SYMMETRIC,    SMC_BIDIRECTIONAL, SMC_SUBSTITUTION ], [3,6] )

  character(len=13), parameter :: STRUCTURE_NAME(0:1) =                        &
      [ character(len=13) :: "nonsymmetric", "symmetric" ]
  character(len=13), parameter :: PARTITION_NAME(0:2) =                        &
      [ character(len=13) :: "column", "row", "bidirectional" ]
  character(len=12), parameter :: DECOMPRESSION_NAME(0:1) =                    &
      [ character(len=12) :: "direct", "substitution" ]
  character(len=21), parameter :: ORDER_NAME(0:4) =                            &
      [ character(len=21) :: "natural", "largest_first", "smallest_last",      &
                             "incidence_degree", "dynamic_largest_first" ]

  ! The value a rejected call must leave in place.
  real(c_double), parameter :: SENTINEL   = -987.0_c_double
  real(c_float),  parameter :: SENTINEL32 = -987.0_c_float

  integer            :: n_pass = 0
  integer            :: n_fail = 0
  character(len=110) :: ctx = ""
  logical            :: verbose = .false.
  character(len=8)   :: verbose_env

  call get_environment_variable("SMC_TEST_VERBOSE", verbose_env)
  verbose = (len_trim(verbose_env) > 0 .and. trim(verbose_env) /= "0")

  call build_matrices()

  call test_version()
  call test_default_options()
  call test_null_options()
  call test_unsupported_combinations()
  call test_invalid_arguments()
  call test_invalid_handle()
  call test_sizing_queries()
  call test_all_combinations()
  call test_index_base()
  call test_postprocessing()

  write(*,*)
  write(*,'(I0,A,I0,A)') n_pass, " checks passed, ", n_fail, " failed"
  if (n_fail > 0) stop 1

contains

  ! Set SMC_TEST_VERBOSE=1 to echo every check as it is reached: a crash inside
  ! the library kills the process without a FAIL line, and then the last line
  ! printed is the last check that completed.
  subroutine check(cond, msg)
    logical,          intent(in) :: cond
    character(len=*), intent(in) :: msg
    if (verbose) then
      if (len_trim(ctx) > 0) then
        write(*,'(A,A,A,A,A)') "  .. ", trim(msg), "  [", trim(ctx), "]"
      else
        write(*,'(A,A)') "  .. ", trim(msg)
      end if
      flush(output_unit)
    end if
    if (cond) then
      n_pass = n_pass + 1
    else
      n_fail = n_fail + 1
      if (len_trim(ctx) > 0) then
        write(*,'(A,A,A,A,A)') "  FAIL  ", trim(msg), "  [", trim(ctx), "]"
      else
        write(*,'(A,A)') "  FAIL  ", trim(msg)
      end if
    end if
  end subroutine check

  ! Test matrices: the same patterns as interfaces/test/C/test_coloring.c.

  subroutine set_matrix(A, name, m, n, colptr, rowval, nzval)
    type(TestMatrix), intent(out) :: A
    character(len=*), intent(in)  :: name
    integer(c_int),   intent(in)  :: m, n
    integer(c_int),   intent(in)  :: colptr(:), rowval(:)
    real(c_double),   intent(in)  :: nzval(:)

    A%name = name
    A%m    = m
    A%n    = n
    A%nnz  = colptr(n+1)          ! 0-based colptr, so colptr(n+1) is the count
    A%colptr = 0
    A%rowval = 0
    A%nzval  = 0.0_c_double
    A%colptr(1:n+1)   = colptr(1:n+1)
    A%rowval(1:A%nnz) = rowval(1:A%nnz)
    A%nzval(1:A%nnz)  = nzval(1:A%nnz)
  end subroutine set_matrix

  ! The same pattern with every index shifted to base 1.
  subroutine shift_matrix(A, B)
    type(TestMatrix), intent(in)  :: A
    type(TestMatrix), intent(out) :: B
    B = A
    B%colptr(1:A%n+1) = A%colptr(1:A%n+1) + 1
    B%rowval(1:A%nnz) = A%rowval(1:A%nnz) + 1
  end subroutine shift_matrix

  subroutine build_matrices()
    integer :: k

    ! 4x6 rectangular, nonsymmetric (the `compress` docstring matrix):
    !   . . 4 6 . 9
    !   1 . . . 7 .
    !   . 2 . . 8 .
    !   . 3 5 . . .
    call set_matrix(NONSYM(1), "nonsym 4x6", 4, 6,                             &
         [ 0, 1, 3, 5, 6, 8, 9 ],                                              &
         [ 1, 2, 3, 0, 3, 0, 1, 2, 0 ],                                        &
         real([ 1, 2, 3, 4, 5, 6, 7, 8, 9 ], c_double))

    ! 7x5 rectangular with denser rows and columns.
    call set_matrix(NONSYM(2), "nonsym 7x5", 7, 5,                             &
         [ 0, 4, 7, 10, 13, 16 ],                                              &
         [ 0, 1, 4, 6,                                                         &
           1, 2, 5,                                                            &
           2, 3, 6,                                                            &
           0, 3, 4,                                                            &
           1, 4, 5 ],                                                          &
         real([ 1, 3, 2, 7,                                                    &
                4, 6, 5,                                                       &
                7, 8, 8,                                                       &
                2, 9, 3,                                                       &
                5, 4, 6 ], c_double))

    ! 7x7 symmetric with a nonzero diagonal: tridiagonal plus the (1,7) corner.
    call set_matrix(SYMM(1), "sym 7x7 full diag", 7, 7,                        &
         [ 0, 3, 6, 9, 12, 15, 18, 21 ],                                       &
         [ 0, 1, 6,                                                            &
           0, 1, 2,                                                            &
           1, 2, 3,                                                            &
           2, 3, 4,                                                            &
           3, 4, 5,                                                            &
           4, 5, 6,                                                            &
           0, 5, 6 ],                                                          &
         real([ 2, 1, 3,                                                       &
                1, 2, 1,                                                       &
                1, 2, 1,                                                       &
                1, 2, 1,                                                       &
                1, 2, 1,                                                       &
                1, 2, 1,                                                       &
                3, 1, 2 ], c_double))

    ! 6x6 symmetric with a zero diagonal (the 3-cube minus a perfect matching);
    ! a zero diagonal is what lets postprocessing hand out the neutral color 0.
    call set_matrix(SYMM(2), "sym 6x6 zero diag", 6, 6,                        &
         [ 0, 2, 4, 6, 8, 10, 12 ],                                            &
         [ 1, 2,                                                               &
           0, 3,                                                               &
           0, 4,                                                               &
           1, 5,                                                               &
           2, 5,                                                               &
           3, 4 ],                                                             &
         real([ 1, 2,                                                          &
                1, 3,                                                          &
                2, 4,                                                          &
                3, 5,                                                          &
                4, 6,                                                          &
                5, 6 ], c_double))

    do k = 1, 2
      call shift_matrix(NONSYM(k), NONSYM1(k))
      call shift_matrix(SYMM(k),   SYMM1(k))
    end do
  end subroutine build_matrices

  ! Column j of a base-0 pattern occupies the Fortran positions
  ! colptr(j)+1 .. colptr(j+1), and rowval(k)+1 is its Fortran row index.
  subroutine build_nz(A, nz)
    type(TestMatrix), intent(in)  :: A
    logical,          intent(out) :: nz(MAXDIM,MAXDIM)
    integer :: i, j, k
    nz = .false.
    do j = 1, A%n
      do k = A%colptr(j) + 1, A%colptr(j+1)
        i = A%rowval(k) + 1
        nz(i,j) = .true.
      end do
    end do
  end subroutine build_nz

  subroutine build_dense(A, dense)
    type(TestMatrix), intent(in)  :: A
    real(c_double),   intent(out) :: dense(MAXDIM,MAXDIM)
    integer :: i, j, k
    dense = 0.0_c_double
    do j = 1, A%n
      do k = A%colptr(j) + 1, A%colptr(j+1)
        i = A%rowval(k) + 1
        dense(i,j) = A%nzval(k)
      end do
    end do
  end subroutine build_dense

  ! Two columns of the same nonzero color must have disjoint row supports.
  logical function column_colors_disjoint(nz, m, n, colors)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: m, n
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer :: i, j, k
    column_colors_disjoint = .true.
    do j = 1, n
      do k = j + 1, n
        if (colors(j) == 0 .or. colors(k) == 0) cycle
        if (colors(j) /= colors(k)) cycle
        do i = 1, m
          if (nz(i,j) .and. nz(i,k)) then
            column_colors_disjoint = .false.
            return
          end if
        end do
      end do
    end do
  end function column_colors_disjoint

  ! The same statement transposed.
  logical function row_colors_disjoint(nz, m, n, colors)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: m, n
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer :: i, j, k
    row_colors_disjoint = .true.
    do i = 1, m
      do k = i + 1, m
        if (colors(i) == 0 .or. colors(k) == 0) cycle
        if (colors(i) /= colors(k)) cycle
        do j = 1, n
          if (nz(i,j) .and. nz(k,j)) then
            row_colors_disjoint = .false.
            return
          end if
        end do
      end do
    end do
  end function row_colors_disjoint

  ! Every nonzero must be readable off the compressed matrix: A(i,j) is either
  ! the only nonzero of row i among the columns of its color, or the only
  ! nonzero of column j among the rows of its color.
  logical function directly_recoverable(nz, m, n, rows, cols, use_rows, use_cols)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: m, n
    integer(c_int), intent(in) :: rows(MAXDIM), cols(MAXDIM)
    logical,        intent(in) :: use_rows, use_cols
    integer :: i, j, k
    logical :: by_column, by_row
    directly_recoverable = .true.
    do j = 1, n
      do i = 1, m
        if (.not. nz(i,j)) cycle
        by_column = .false.
        by_row    = .false.
        if (use_cols) then
          if (cols(j) /= 0) then
            by_column = .true.
            do k = 1, n
              if (k /= j .and. nz(i,k) .and. cols(k) == cols(j)) by_column = .false.
            end do
          end if
        end if
        if (use_rows) then
          if (rows(i) /= 0) then
            by_row = .true.
            do k = 1, m
              if (k /= i .and. nz(k,j) .and. rows(k) == rows(i)) by_row = .false.
            end do
          end if
        end if
        if (.not. by_column .and. .not. by_row) then
          directly_recoverable = .false.
          return
        end if
      end do
    end do
  end function directly_recoverable

  ! Adjacent vertices carry different nonzero colors.
  logical function is_proper(nz, n, colors)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: n
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer :: i, j
    is_proper = .true.
    do j = 1, n
      do i = 1, j - 1
        if (.not. nz(i,j)) cycle
        if (colors(i) == 0 .or. colors(j) == 0) cycle
        if (colors(i) == colors(j)) then
          is_proper = .false.
          return
        end if
      end do
    end do
  end function is_proper

  ! No path i - j - k - l uses only two colors (a star coloring).
  logical function is_star_coloring(nz, n, colors)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: n
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer :: i, j, k, l
    is_star_coloring = .true.
    do j = 1, n
      do k = 1, n
        if (j == k) cycle
        if (.not. nz(j,k)) cycle
        do i = 1, n
          if (i == j .or. i == k) cycle
          if (.not. nz(i,j)) cycle
          do l = 1, n
            if (l == i .or. l == j .or. l == k) cycle
            if (.not. nz(k,l)) cycle
            if (colors(i) == 0 .or. colors(j) == 0) cycle
            if (colors(k) == 0 .or. colors(l) == 0) cycle
            if (colors(i) == colors(k) .and. colors(j) == colors(l)) then
              is_star_coloring = .false.
              return
            end if
          end do
        end do
      end do
    end do
  end function is_star_coloring

  integer function uf_find(parent, x)
    integer, intent(inout) :: parent(MAXDIM)
    integer, intent(in)    :: x
    integer :: r
    r = x
    do while (parent(r) /= r)
      parent(r) = parent(parent(r))
      r = parent(r)
    end do
    uf_find = r
  end function uf_find

  ! Every subgraph induced by two colors is a forest (an acyclic coloring).
  logical function is_acyclic_coloring(nz, n, colors, ncolors)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: n
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer(c_int), intent(in) :: ncolors
    integer :: ca, cb, i, j, ri, rj, parent(MAXDIM)
    is_acyclic_coloring = .true.
    do ca = 1, ncolors
      do cb = ca + 1, ncolors
        do i = 1, MAXDIM
          parent(i) = i
        end do
        do j = 1, n
          do i = 1, j - 1
            if (.not. nz(i,j)) cycle
            if (.not. ((colors(i) == ca .and. colors(j) == cb) .or.            &
                       (colors(i) == cb .and. colors(j) == ca))) cycle
            ri = uf_find(parent, i)
            rj = uf_find(parent, j)
            if (ri == rj) then           ! a cycle inside {ca, cb}
              is_acyclic_coloring = .false.
              return
            end if
            parent(ri) = rj
          end do
        end do
      end do
    end do
  end function is_acyclic_coloring

  ! Symmetric counterpart: A(i,j) is read from B(i, colors(j)) when column j is
  ! the only one of its color meeting row i, or - by symmetry - B(j, colors(i)).
  logical function symmetrically_recoverable(nz, n, colors)
    logical,        intent(in) :: nz(MAXDIM,MAXDIM)
    integer(c_int), intent(in) :: n
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer :: i, j, k
    logical :: by_j, by_i
    symmetrically_recoverable = .true.
    do j = 1, n
      do i = 1, n
        if (.not. nz(i,j)) cycle
        by_j = .false.
        by_i = .false.
        if (colors(j) /= 0) then
          by_j = .true.
          do k = 1, n
            if (k /= j .and. nz(i,k) .and. colors(k) == colors(j)) by_j = .false.
          end do
        end if
        if (colors(i) /= 0) then
          by_i = .true.
          do k = 1, n
            if (k /= i .and. nz(j,k) .and. colors(k) == colors(i)) by_i = .false.
          end do
        end if
        if (.not. by_j .and. .not. by_i) then
          symmetrically_recoverable = .false.
          return
        end if
      end do
    end do
  end function symmetrically_recoverable

  integer(c_int) function color_matrix(A, o, result)
    type(TestMatrix),         target, intent(in)  :: A
    type(SmcColoringOptions), target, intent(in)  :: o
    type(c_ptr),                      intent(out) :: result
    color_matrix = smc_coloring(A%m, A%n, c_loc(A%colptr), c_loc(A%rowval),    &
                                c_loc(o), result)
  end function color_matrix

  ! The groups are exactly the color classes.
  logical function groups_match_colors(result, ngroups, colors, len, base, column)
    type(c_ptr),    intent(in) :: result
    integer(c_int), intent(in) :: ngroups
    integer(c_int), intent(in) :: colors(MAXDIM)
    integer(c_int), intent(in) :: len, base
    logical,        intent(in) :: column

    integer(c_int), target :: members(MAXDIM), gsize
    integer(c_int) :: ret
    integer        :: g, k, index
    logical        :: seen(MAXDIM)

    groups_match_colors = .true.
    seen = .false.

    do g = 1, ngroups
      gsize = -1
      if (column) then
        ret = smc_column_group_size(result, int(g, c_int), c_loc(gsize))
      else
        ret = smc_row_group_size(result, int(g, c_int), c_loc(gsize))
      end if
      if (ret /= 0 .or. gsize < 0 .or. gsize > len) then
        groups_match_colors = .false.
        return
      end if
      if (column) then
        ret = smc_column_group(result, int(g, c_int), c_loc(members), gsize)
      else
        ret = smc_row_group(result, int(g, c_int), c_loc(members), gsize)
      end if
      if (ret /= 0) then
        groups_match_colors = .false.
        return
      end if
      do k = 1, gsize
        ! members are in the caller's index base; make them Fortran indices.
        index = members(k) - base + 1
        if (index < 1 .or. index > len) then
          groups_match_colors = .false.
          return
        end if
        if (seen(index)) then                 ! a member of two groups
          groups_match_colors = .false.
          return
        end if
        seen(index) = .true.
        if (colors(index) /= g) then          ! wrong group
          groups_match_colors = .false.
          return
        end if
      end do
    end do

    ! Every non-neutral index belongs to exactly one group, and only those.
    do k = 1, len
      if (colors(k) /= 0 .and. .not. seen(k)) groups_match_colors = .false.
      if (colors(k) == 0 .and. seen(k))       groups_match_colors = .false.
      if (colors(k) < 0 .or. colors(k) > ngroups) groups_match_colors = .false.
    end do
  end function groups_match_colors

  ! The buffer lengths, read back from the API rather than from TestMatrix,
  ! which also checks that smc_nnz and smc_size agree with the colored pattern.
  ! ok is .false. if any query fails or disagrees.
  subroutine query_lengths(result, A, Br_rows, Br_cols, Bc_rows, Bc_cols,      &
                           br_len, bc_len, a_len, nz_len, ok)
    type(c_ptr),       intent(in)  :: result
    type(TestMatrix),  intent(in)  :: A
    integer(c_int),    intent(out) :: Br_rows, Br_cols, Bc_rows, Bc_cols
    integer(c_size_t), intent(out) :: br_len, bc_len, a_len, nz_len
    logical,           intent(out) :: ok

    integer(c_int), target :: brr, brc, bcr, bcc, nnz_q, m_q, n_q
    integer(c_int) :: ret

    Br_rows = 0; Br_cols = 0; Bc_rows = 0; Bc_cols = 0
    br_len = 0; bc_len = 0; a_len = 0; nz_len = 0
    ok = .false.

    brr = -1; brc = -1; bcr = -1; bcc = -1
    ret = smc_compressed_size(result, c_loc(brr), c_loc(brc), c_loc(bcr), c_loc(bcc))
    if (ret /= 0) return

    nnz_q = -1
    ret = smc_nnz(result, c_loc(nnz_q))
    if (ret /= 0 .or. nnz_q /= A%nnz) return

    m_q = -1; n_q = -1
    ret = smc_size(result, c_loc(m_q), c_loc(n_q))
    if (ret /= 0 .or. m_q /= A%m .or. n_q /= A%n) return

    Br_rows = brr; Br_cols = brc; Bc_rows = bcr; Bc_cols = bcc
    br_len = int(brr, c_size_t) * int(brc, c_size_t)
    bc_len = int(bcr, c_size_t) * int(bcc, c_size_t)
    a_len  = int(m_q, c_size_t) * int(n_q, c_size_t)
    nz_len = int(nnz_q, c_size_t)
    ok = .true.
  end subroutine query_lengths

  logical function roundtrip_f64(result, A, exact)
    type(c_ptr),                intent(in) :: result
    type(TestMatrix),   target, intent(in) :: A
    logical,                    intent(in) :: exact

    integer(c_int)    :: Br_rows, Br_cols, Bc_rows, Bc_cols, ret
    integer(c_size_t) :: br_len, bc_len, a_len, nz_len
    logical           :: ok, bidir
    real(c_double), target, allocatable :: Br(:), Bc(:), out(:)
    real(c_double)    :: dense(MAXDIM,MAXDIM), expected, got
    type(c_ptr)       :: Br_ptr
    integer           :: i, j

    call query_lengths(result, A, Br_rows, Br_cols, Bc_rows, Bc_cols,          &
                       br_len, bc_len, a_len, nz_len, ok)
    roundtrip_f64 = ok
    if (.not. ok) return

    ! Br_cols identifies the partition; br_len does not (it can be 0 for a
    ! bidirectional result whose row coloring was emptied by postprocessing).
    bidir = (Br_cols > 0)

    ! c_loc needs a nonzero-sized object, so never allocate 0 elements.
    allocate(Br(max(br_len, 1_c_size_t)))
    allocate(Bc(max(bc_len, 1_c_size_t)))
    allocate(out(max(a_len,  1_c_size_t)))
    Br = SENTINEL
    Bc = SENTINEL
    out = SENTINEL

    if (bidir) then
      Br_ptr = c_loc(Br)
    else
      Br_ptr = c_null_ptr          ! the documented call for the other partitions
    end if

    ret = smc_compress(result, c_loc(A%nzval), nz_len, Br_ptr, br_len, c_loc(Bc), bc_len)
    if (ret /= 0) roundtrip_f64 = .false.
    ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len, c_loc(out), a_len)
    if (ret /= 0) roundtrip_f64 = .false.

    if (roundtrip_f64) then
      call build_dense(A, dense)
      do j = 1, A%n
        do i = 1, A%m
          expected = dense(i,j)
          got = out((j-1)*A%m + i)
          if (exact) then
            if (got /= expected) roundtrip_f64 = .false.
          else
            if (abs(got - expected) > 1.0e-9_c_double * (1.0_c_double + abs(expected))) &
              roundtrip_f64 = .false.
          end if
        end do
      end do
    end if

    deallocate(Br, Bc, out)
  end function roundtrip_f64

  logical function roundtrip_f32(result, A, exact)
    type(c_ptr),                intent(in) :: result
    type(TestMatrix),           intent(in) :: A
    logical,                    intent(in) :: exact

    integer(c_int)    :: Br_rows, Br_cols, Bc_rows, Bc_cols, ret
    integer(c_size_t) :: br_len, bc_len, a_len, nz_len
    logical           :: ok, bidir
    real(c_float), target, allocatable :: Br(:), Bc(:), out(:)
    real(c_float), target :: nz32(MAXNNZ)
    real(c_double)    :: dense(MAXDIM,MAXDIM)
    real(c_float)     :: expected, got
    type(c_ptr)       :: Br_ptr
    integer           :: i, j

    call query_lengths(result, A, Br_rows, Br_cols, Bc_rows, Bc_cols,          &
                       br_len, bc_len, a_len, nz_len, ok)
    roundtrip_f32 = ok
    if (.not. ok) return

    bidir = (Br_cols > 0)

    ! Lengths are element counts, so they match the Float64 numbers.
    nz32 = 0.0_c_float
    nz32(1:A%nnz) = real(A%nzval(1:A%nnz), c_float)

    allocate(Br(max(br_len, 1_c_size_t)))
    allocate(Bc(max(bc_len, 1_c_size_t)))
    allocate(out(max(a_len,  1_c_size_t)))
    Br = SENTINEL32
    Bc = SENTINEL32
    out = SENTINEL32

    if (bidir) then
      Br_ptr = c_loc(Br)
    else
      Br_ptr = c_null_ptr
    end if

    ret = smc_compress(result, c_loc(nz32), nz_len, Br_ptr, br_len, c_loc(Bc), bc_len)
    if (ret /= 0) roundtrip_f32 = .false.
    ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len, c_loc(out), a_len)
    if (ret /= 0) roundtrip_f32 = .false.

    if (roundtrip_f32) then
      call build_dense(A, dense)
      do j = 1, A%n
        do i = 1, A%m
          expected = real(dense(i,j), c_float)
          got = out((j-1)*A%m + i)
          if (exact) then
            if (got /= expected) roundtrip_f32 = .false.
          else
            if (abs(got - expected) > 1.0e-4_c_float * (1.0_c_float + abs(expected))) &
              roundtrip_f32 = .false.
          end if
        end do
      end do
    end if

    deallocate(Br, Bc, out)
  end function roundtrip_f32

  ! Every buffer length, one element short, one at a time.  Each buffer keeps
  ! its full size and is pre-filled, so a -3 can only come from the length check
  ! and a surviving sentinel proves the check ran before any write.  Lengths are
  ! only understated when positive; see pitfall 2 in the file header.
  logical function length_guards(result, A)
    type(c_ptr),              intent(in) :: result
    type(TestMatrix), target, intent(in) :: A

    integer(c_int)    :: Br_rows, Br_cols, Bc_rows, Bc_cols, ret
    integer(c_size_t) :: br_len, bc_len, a_len, nz_len
    logical           :: ok, bidir
    real(c_double), target, allocatable :: Br(:), Bc(:), out(:)
    type(c_ptr)       :: Br_ptr
    integer(c_size_t) :: k

    call query_lengths(result, A, Br_rows, Br_cols, Bc_rows, Bc_cols,          &
                       br_len, bc_len, a_len, nz_len, ok)
    length_guards = ok
    if (.not. ok) return

    bidir = (Br_cols > 0)

    allocate(Br(max(br_len, 1_c_size_t)))
    allocate(Bc(max(bc_len, 1_c_size_t)))
    allocate(out(max(a_len,  1_c_size_t)))

    if (bidir) then
      Br_ptr = c_loc(Br)
    else
      Br_ptr = c_null_ptr
    end if

    ! The exact sizes work, and fill Bc / Br with something decompress can use.
    ret = smc_compress(result, c_loc(A%nzval), nz_len, Br_ptr, br_len, c_loc(Bc), bc_len)
    if (ret /= 0) length_guards = .false.
    ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len, c_loc(out), a_len)
    if (ret /= 0) length_guards = .false.

    ! ---- smc_compress -----------------------------------------------------
    if (nz_len > 0) then
      ret = smc_compress(result, c_loc(A%nzval), nz_len - 1, Br_ptr, br_len,   &
                         c_loc(Bc), bc_len)
      if (ret /= -3) length_guards = .false.
    end if
    if (bc_len > 0) then
      ret = smc_compress(result, c_loc(A%nzval), nz_len, Br_ptr, br_len,       &
                         c_loc(Bc), bc_len - 1)
      if (ret /= -3) length_guards = .false.
    end if
    ret = smc_compress(result, c_null_ptr, nz_len, Br_ptr, br_len, c_loc(Bc), bc_len)
    if (ret /= -3) length_guards = .false.
    ret = smc_compress(result, c_loc(A%nzval), nz_len, Br_ptr, br_len, c_null_ptr, bc_len)
    if (ret /= -3) length_guards = .false.

    if (bidir) then
      if (br_len > 0) then
        ret = smc_compress(result, c_loc(A%nzval), nz_len, c_loc(Br), br_len - 1, &
                           c_loc(Bc), bc_len)
        if (ret /= -3) length_guards = .false.
      end if
      ! Both compressed matrices are required.
      ret = smc_compress(result, c_loc(A%nzval), nz_len, c_null_ptr, 0_c_size_t, &
                         c_loc(Bc), bc_len)
      if (ret /= -3) length_guards = .false.
    else
      ! Br is unused: c_null_ptr with a length of 0 is the documented call.
      ret = smc_compress(result, c_loc(A%nzval), nz_len, c_null_ptr, 0_c_size_t, &
                         c_loc(Bc), bc_len)
      if (ret /= 0) length_guards = .false.
    end if

    ! A generous length is a promise, not an error: the comparison is unsigned
    ! on the C side and must not wrap.
    ret = smc_compress(result, c_loc(A%nzval), huge(0_c_size_t), Br_ptr, br_len, &
                       c_loc(Bc), bc_len)
    if (ret /= 0) length_guards = .false.

    ret = smc_compress(result, c_loc(A%nzval), nz_len, Br_ptr, br_len, c_loc(Bc), bc_len)
    if (ret /= 0) length_guards = .false.

    ! ---- smc_decompress ---------------------------------------------------
    out = SENTINEL
    if (a_len > 0) then
      ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len, c_loc(out), a_len - 1)
      if (ret /= -3) length_guards = .false.
    end if
    if (bc_len > 0) then
      ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len - 1, c_loc(out), a_len)
      if (ret /= -3) length_guards = .false.
    end if
    if (bidir .and. br_len > 0) then
      ret = smc_decompress(result, c_loc(Br), br_len - 1, c_loc(Bc), bc_len, c_loc(out), a_len)
      if (ret /= -3) length_guards = .false.
    end if
    if (bidir) then
      ret = smc_decompress(result, c_null_ptr, 0_c_size_t, c_loc(Bc), bc_len,  &
                           c_loc(out), a_len)
      if (ret /= -3) length_guards = .false.
    end if
    ret = smc_decompress(result, Br_ptr, br_len, c_null_ptr, bc_len, c_loc(out), a_len)
    if (ret /= -3) length_guards = .false.
    ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len, c_null_ptr, a_len)
    if (ret /= -3) length_guards = .false.

    do k = 1, a_len
      if (out(k) /= SENTINEL) length_guards = .false.
    end do

    ! The exact sizes still work after every rejection.
    ret = smc_decompress(result, Br_ptr, br_len, c_loc(Bc), bc_len, c_loc(out), a_len)
    if (ret /= 0) length_guards = .false.

    deallocate(Br, Bc, out)
  end function length_guards

  subroutine test_case(A, o, label)
    type(TestMatrix),         target, intent(in) :: A
    type(SmcColoringOptions),         intent(in) :: o
    character(len=*),                 intent(in) :: label

    type(SmcColoringOptions), target :: opts
    type(c_ptr)    :: result
    integer(c_int), target :: ncolors, ngroups_c, ngroups_r
    integer(c_int), target :: colors_c(MAXDIM), colors_r(MAXDIM)
    integer(c_int) :: ret
    logical        :: has_columns, has_rows, direct
    logical        :: nz(MAXDIM,MAXDIM)

    opts = o
    ctx = A%name(1:len_trim(A%name)) // " / " // label(1:len_trim(label))

    result = c_null_ptr
    ret = color_matrix(A, opts, result)
    if (ret /= 0) then
      call check(.false., "smc_coloring succeeds")
      ctx = ""
      return
    end if

    has_columns = (opts%partition /= SMC_ROW)
    has_rows    = (opts%partition /= SMC_COLUMN)
    direct      = (opts%decompression == SMC_DIRECT)
    call build_nz(A, nz)

    colors_c = 0
    colors_r = 0
    ncolors = 0
    ngroups_c = 0
    ngroups_r = 0

    ret = smc_ncolors(result, c_loc(ncolors))
    call check(ret == 0 .and. ncolors > 0, "ncolors is positive")

    if (has_columns) then
      ret = smc_column_colors(result, c_loc(colors_c), A%n)
      call check(ret == 0, "smc_column_colors succeeds")
      ret = smc_ncolumn_groups(result, c_loc(ngroups_c))
      call check(ret == 0, "smc_ncolumn_groups succeeds")
      call check(groups_match_colors(result, ngroups_c, colors_c, A%n,         &
                                     opts%index_base, .true.),                 &
                 "column groups are exactly the color classes")
    else
      ret = smc_column_colors(result, c_loc(colors_c), A%n)
      call check(ret == -2, "a row partition has no column coloring (-2)")
    end if

    if (has_rows) then
      ret = smc_row_colors(result, c_loc(colors_r), A%m)
      call check(ret == 0, "smc_row_colors succeeds")
      ret = smc_nrow_groups(result, c_loc(ngroups_r))
      call check(ret == 0, "smc_nrow_groups succeeds")
      call check(groups_match_colors(result, ngroups_r, colors_r, A%m,         &
                                     opts%index_base, .false.),                &
                 "row groups are exactly the color classes")
    else
      ret = smc_row_colors(result, c_loc(colors_r), A%m)
      call check(ret == -2, "a column partition has no row coloring (-2)")
    end if

    if (opts%partition == SMC_BIDIRECTIONAL) then
      call check(ncolors == ngroups_r + ngroups_c, "ncolors counts both dimensions")
    else if (has_columns) then
      call check(ncolors == ngroups_c, "ncolors is the number of column groups")
    else
      call check(ncolors == ngroups_r, "ncolors is the number of row groups")
    end if

    ! ---- structural validity, from the pattern alone ----------------------
    if (opts%structure == SMC_NONSYMMETRIC .and. opts%partition == SMC_COLUMN) then
      call check(column_colors_disjoint(nz, A%m, A%n, colors_c),               &
                 "columns of one color share no nonzero row")
      call check(directly_recoverable(nz, A%m, A%n, colors_r, colors_c, .false., .true.), &
                 "every nonzero is recoverable from its column group")
    else if (opts%structure == SMC_NONSYMMETRIC .and. opts%partition == SMC_ROW) then
      call check(row_colors_disjoint(nz, A%m, A%n, colors_r),                  &
                 "rows of one color share no nonzero column")
      call check(directly_recoverable(nz, A%m, A%n, colors_r, colors_c, .true., .false.), &
                 "every nonzero is recoverable from its row group")
    else if (opts%structure == SMC_SYMMETRIC) then
      call check(is_proper(nz, A%n, colors_c), "the symmetric coloring is proper")
      if (direct) then
        call check(is_star_coloring(nz, A%n, colors_c),                        &
                   "the symmetric coloring is a star coloring")
        call check(symmetrically_recoverable(nz, A%n, colors_c),               &
                   "every nonzero is recoverable, using the symmetry of A")
      else
        call check(is_acyclic_coloring(nz, A%n, colors_c, ncolors),            &
                   "the symmetric coloring is acyclic")
      end if
    else if (direct) then                ! nonsymmetric bidirectional, direct
      call check(directly_recoverable(nz, A%m, A%n, colors_r, colors_c, .true., .true.), &
                 "every nonzero is recoverable from a row or a column group")
    end if

    ! ---- the compressed form determines the matrix ------------------------
    if (opts%dtype == SMC_FLOAT32) then
      call check(roundtrip_f32(result, A, direct),                             &
                 "Float32 compress -> decompress round trip")
    else
      call check(roundtrip_f64(result, A, direct),                             &
                 "Float64 compress -> decompress round trip")
      ! The length checks count elements, so one Float64 pass covers them.
      call check(length_guards(result, A),                                     &
                 "every buffer length is checked, and short is -3")
    end if

    ret = smc_result_free(result)
    call check(ret == 0, "smc_result_free succeeds")
    ctx = ""
  end subroutine test_case

  subroutine test_fast_coloring(A, o)
    type(TestMatrix),         target, intent(in) :: A
    type(SmcColoringOptions),         intent(in) :: o

    type(SmcColoringOptions), target :: opts
    type(c_ptr)    :: result
    integer(c_int), target :: rows(MAXDIM), cols(MAXDIM)
    integer(c_int), target :: frows(MAXDIM), fcols(MAXDIM)
    integer(c_int), target :: ncolors, fncolors
    integer(c_int) :: ret, want
    integer        :: k
    logical        :: same

    opts = o
    ctx = A%name(1:len_trim(A%name)) // " / fast_coloring"

    rows = 0; cols = 0; frows = 0; fcols = 0
    ncolors = 0

    result = c_null_ptr
    ret = color_matrix(A, opts, result)
    if (ret /= 0) then
      call check(.false., "smc_coloring succeeds in the fast_coloring comparison")
      ctx = ""
      return
    end if
    ret = smc_ncolors(result, c_loc(ncolors))
    if (opts%partition /= SMC_ROW)    ret = smc_column_colors(result, c_loc(cols), A%n)
    if (opts%partition /= SMC_COLUMN) ret = smc_row_colors(result, c_loc(rows), A%m)
    ret = smc_result_free(result)
    call check(ret == 0, "free before the fast_coloring comparison")

    fncolors = -1
    ret = smc_fast_coloring(A%m, A%n, c_loc(A%colptr), c_loc(A%rowval), c_loc(opts), &
                            c_loc(frows), c_loc(fcols), c_loc(fncolors))
    call check(ret == 0, "smc_fast_coloring succeeds")
    call check(fncolors == ncolors, "smc_fast_coloring reports the same number of colors")

    same = .true.
    if (opts%partition /= SMC_ROW) then
      do k = 1, A%n
        if (fcols(k) /= cols(k)) same = .false.
      end do
    end if
    if (opts%partition /= SMC_COLUMN) then
      do k = 1, A%m
        if (frows(k) /= rows(k)) same = .false.
      end do
    end if
    call check(same, "smc_fast_coloring agrees with smc_coloring")

    ! A buffer may be c_null_ptr exactly when the partition produces no coloring
    ! for that dimension.
    fncolors = -1
    if (opts%partition == SMC_COLUMN) then
      want = 0
    else
      want = -3
    end if
    ret = smc_fast_coloring(A%m, A%n, c_loc(A%colptr), c_loc(A%rowval), c_loc(opts), &
                            c_null_ptr, c_loc(fcols), c_loc(fncolors))
    call check(ret == want, "c_null_ptr row_colors")

    fncolors = -1
    if (opts%partition == SMC_ROW) then
      want = 0
    else
      want = -3
    end if
    ret = smc_fast_coloring(A%m, A%n, c_loc(A%colptr), c_loc(A%rowval), c_loc(opts), &
                            c_loc(frows), c_null_ptr, c_loc(fncolors))
    call check(ret == want, "c_null_ptr column_colors")
    ctx = ""
  end subroutine test_fast_coloring

  subroutine test_all_combinations()
    integer :: c, order, post, dt, k
    type(SmcColoringOptions) :: o
    character(len=80) :: label
    character(len=13) :: sname, pname
    character(len=12) :: dname
    character(len=21) :: oname

    do c = 1, 6
      sname = STRUCTURE_NAME(SUPPORTED(1,c))
      pname = PARTITION_NAME(SUPPORTED(2,c))
      dname = DECOMPRESSION_NAME(SUPPORTED(3,c))
      write(*,'(A,A,A,A,A,A)') sname(1:len_trim(sname)), " / ",                &
                               pname(1:len_trim(pname)), " / ",                &
                               dname(1:len_trim(dname)), " ..."
      do order = 0, 4
        do post = 0, 1
          do dt = 0, 1
            do k = 1, 2
              o = smc_default_options()
              o%structure     = SUPPORTED(1,c)
              o%partition     = SUPPORTED(2,c)
              o%decompression = SUPPORTED(3,c)
              o%order         = int(order, c_int)
              o%postprocessing = int(post, c_int)
              o%dtype         = int(dt, c_int)
              oname = ORDER_NAME(order)
              write(label,'(A,A,A,I0,A,I0)') "order=", oname(1:len_trim(oname)),&
                                             " postprocessing=", post, " dtype=", dt
              if (SUPPORTED(1,c) == SMC_SYMMETRIC) then
                call test_case(SYMM(k), o, label)
              else
                call test_case(NONSYM(k), o, label)
              end if
            end do
          end do
        end do
      end do

      ! fast_coloring only needs one pass per combination.
      do k = 1, 2
        o = smc_default_options()
        o%structure     = SUPPORTED(1,c)
        o%partition     = SUPPORTED(2,c)
        o%decompression = SUPPORTED(3,c)
        o%order         = SMC_LARGEST_FIRST
        if (SUPPORTED(1,c) == SMC_SYMMETRIC) then
          call test_fast_coloring(SYMM(k), o)
        else
          call test_fast_coloring(NONSYM(k), o)
        end if
      end do
    end do
  end subroutine test_all_combinations

  ! index_base 0 and 1 describe the same matrix, hence the same coloring; only
  ! the group members are shifted.
  subroutine test_index_base()
    integer :: c, k, j, g
    type(SmcColoringOptions), target :: o0, o1
    type(c_ptr)    :: r0, r1
    integer(c_int), target :: c0(MAXDIM), c1(MAXDIM)
    integer(c_int), target :: nc0, nc1, g0, g1, s0, s1
    integer(c_int), target :: m0(MAXDIM), m1(MAXDIM)
    integer(c_int) :: ret, mm, nn
    logical :: same, shifted

    write(*,'(A)') "index_base 0 vs 1 ..."
    do c = 1, 6
      do k = 1, 2
        o0 = smc_default_options()
        o1 = smc_default_options()
        o0%structure     = SUPPORTED(1,c)
        o1%structure     = SUPPORTED(1,c)
        o0%partition     = SUPPORTED(2,c)
        o1%partition     = SUPPORTED(2,c)
        o0%decompression = SUPPORTED(3,c)
        o1%decompression = SUPPORTED(3,c)
        o1%index_base    = 1

        r0 = c_null_ptr
        r1 = c_null_ptr
        if (SUPPORTED(1,c) == SMC_SYMMETRIC) then
          ctx = SYMM(k)%name(1:len_trim(SYMM(k)%name)) // " / index_base"
          ret = color_matrix(SYMM(k),  o0, r0)
          call check(ret == 0, "0-based coloring")
          ret = color_matrix(SYMM1(k), o1, r1)
          call check(ret == 0, "1-based coloring")
          mm = SYMM(k)%m
          nn = SYMM(k)%n
        else
          ctx = NONSYM(k)%name(1:len_trim(NONSYM(k)%name)) // " / index_base"
          ret = color_matrix(NONSYM(k),  o0, r0)
          call check(ret == 0, "0-based coloring")
          ret = color_matrix(NONSYM1(k), o1, r1)
          call check(ret == 0, "1-based coloring")
          mm = NONSYM(k)%m
          nn = NONSYM(k)%n
        end if

        nc0 = 0
        nc1 = 0
        ret = smc_ncolors(r0, c_loc(nc0))
        ret = smc_ncolors(r1, c_loc(nc1))
        call check(nc0 == nc1 .and. nc0 > 0, "index_base does not change the number of colors")

        same = .true.
        if (o0%partition /= SMC_ROW) then
          c0 = 0
          c1 = 0
          ret = smc_column_colors(r0, c_loc(c0), nn)
          ret = smc_column_colors(r1, c_loc(c1), nn)
          do j = 1, nn
            if (c0(j) /= c1(j)) same = .false.
          end do
        end if
        if (o0%partition /= SMC_COLUMN) then
          c0 = 0
          c1 = 0
          ret = smc_row_colors(r0, c_loc(c0), mm)
          ret = smc_row_colors(r1, c_loc(c1), mm)
          do j = 1, mm
            if (c0(j) /= c1(j)) same = .false.
          end do
        end if
        call check(same, "index_base does not change the colors (colors are labels)")

        shifted = .true.
        if (o0%partition /= SMC_ROW) then
          g0 = 0
          g1 = 0
          ret = smc_ncolumn_groups(r0, c_loc(g0))
          ret = smc_ncolumn_groups(r1, c_loc(g1))
          call check(g0 == g1, "same number of column groups")
          do g = 1, min(g0, g1)
            s0 = 0
            s1 = 0
            ret = smc_column_group_size(r0, int(g, c_int), c_loc(s0))
            if (ret /= 0) shifted = .false.
            ret = smc_column_group_size(r1, int(g, c_int), c_loc(s1))
            if (ret /= 0) shifted = .false.
            if (s0 /= s1) then
              shifted = .false.
            else
              m0 = 0
              m1 = 0
              ret = smc_column_group(r0, int(g, c_int), c_loc(m0), s0)
              if (ret /= 0) shifted = .false.
              ret = smc_column_group(r1, int(g, c_int), c_loc(m1), s1)
              if (ret /= 0) shifted = .false.
              do j = 1, s0
                if (m1(j) /= m0(j) + 1) shifted = .false.
              end do
            end if
          end do
        end if
        if (o0%partition /= SMC_COLUMN) then
          g0 = 0
          g1 = 0
          ret = smc_nrow_groups(r0, c_loc(g0))
          ret = smc_nrow_groups(r1, c_loc(g1))
          call check(g0 == g1, "same number of row groups")
          do g = 1, min(g0, g1)
            s0 = 0
            s1 = 0
            ret = smc_row_group_size(r0, int(g, c_int), c_loc(s0))
            if (ret /= 0) shifted = .false.
            ret = smc_row_group_size(r1, int(g, c_int), c_loc(s1))
            if (ret /= 0) shifted = .false.
            if (s0 /= s1) then
              shifted = .false.
            else
              m0 = 0
              m1 = 0
              ret = smc_row_group(r0, int(g, c_int), c_loc(m0), s0)
              if (ret /= 0) shifted = .false.
              ret = smc_row_group(r1, int(g, c_int), c_loc(m1), s1)
              if (ret /= 0) shifted = .false.
              do j = 1, s0
                if (m1(j) /= m0(j) + 1) shifted = .false.
              end do
            end if
          end do
        end if
        call check(shifted, "group members are shifted by exactly the index base")

        ret = smc_result_free(r0)
        call check(ret == 0, "free the 0-based handle")
        ret = smc_result_free(r1)
        call check(ret == 0, "free the 1-based handle")
        ctx = ""
      end do
    end do
  end subroutine test_index_base

  ! postprocessing may only replace colors by the neutral color 0, never make
  ! the coloring worse.
  subroutine test_postprocessing()
    integer :: k, j, l
    type(SmcColoringOptions), target :: off, on
    type(c_ptr) :: r_off, r_on
    integer(c_int), target :: c_off(MAXDIM), c_on(MAXDIM), nc_off, nc_on
    integer(c_int) :: ret, nn
    logical :: valid, injective

    write(*,'(A)') "postprocessing ..."
    do k = 1, 2
      ctx = SYMM(k)%name(1:len_trim(SYMM(k)%name)) // " / postprocessing"
      off = smc_default_options()
      on  = smc_default_options()
      off%structure = SMC_SYMMETRIC
      on%structure  = SMC_SYMMETRIC
      on%postprocessing = 1

      r_off = c_null_ptr
      r_on  = c_null_ptr
      ret = color_matrix(SYMM(k), off, r_off)
      call check(ret == 0, "coloring")
      ret = color_matrix(SYMM(k), on, r_on)
      call check(ret == 0, "coloring (postprocessed)")

      nn = SYMM(k)%n
      nc_off = 0
      nc_on  = 0
      c_off = 0
      c_on  = 0
      ret = smc_ncolors(r_off, c_loc(nc_off))
      ret = smc_ncolors(r_on,  c_loc(nc_on))
      ret = smc_column_colors(r_off, c_loc(c_off), nn)
      ret = smc_column_colors(r_on,  c_loc(c_on),  nn)

      call check(nc_on <= nc_off, "postprocessing never increases the number of colors")

      valid = .true.
      do j = 1, nn
        if (c_on(j) < 0 .or. c_on(j) > nc_on)    valid = .false.
        if (c_off(j) < 1 .or. c_off(j) > nc_off) valid = .false.
      end do
      call check(valid, "colors stay in 0..ncolors, and are nonzero without postprocessing")

      ! Surviving colors are renamed injectively: two vertices keep the same
      ! nonzero color together.
      injective = .true.
      do j = 1, nn
        do l = j + 1, nn
          if (c_on(j) == 0 .or. c_on(l) == 0) cycle
          if ((c_on(j) == c_on(l)) .neqv. (c_off(j) == c_off(l))) injective = .false.
        end do
      end do
      call check(injective, "postprocessing renames colors injectively")

      ret = smc_result_free(r_off)
      call check(ret == 0, "free")
      ret = smc_result_free(r_on)
      call check(ret == 0, "free (postprocessed)")
      ctx = ""
    end do
  end subroutine test_postprocessing

  subroutine test_version()
    ! No `target` needed: smc_version's outputs are bound as intent(out)
    ! scalars, not type(c_ptr), so they are passed by reference directly.
    integer(c_int) :: major, minor, patch
    major = -1
    minor = -1
    patch = -1
    write(*,'(A)') "version ..."
    call smc_version(major, minor, patch)
    call check(major == SMC_VERSION_MAJOR .and. minor == SMC_VERSION_MINOR .and. &
               patch == SMC_VERSION_PATCH,                                       &
               "smc_version matches the SMC_VERSION_* parameters")
  end subroutine test_version

  subroutine test_default_options()
    type(SmcColoringOptions) :: o
    integer :: nonzero_before

    write(*,'(A)') "default options ..."

    ! Every default is 0, so comparing the fields cannot tell a working
    ! smc_default_options from one that writes nothing.  It is also the only
    ! entry point returning a derived type by value, so poison the struct first
    ! and require the call to overwrite it: that exercises the struct-return ABI.
    o%structure         = 111
    o%partition         = 112
    o%decompression     = 113
    o%order             = 114
    o%postprocessing    = 115
    o%symmetric_pattern = 116
    o%index_base        = 117
    o%dtype             = 118
    nonzero_before = o%structure + o%dtype

    o = smc_default_options()

    call check(nonzero_before == 229,                                          &
               "the poison values were actually stored (sanity)")
    call check(o%structure /= 111 .and. o%dtype /= 118,                        &
               "smc_default_options overwrites the struct (struct-return ABI)")
    call check(o%structure     == SMC_NONSYMMETRIC, "default structure is SMC_NONSYMMETRIC")
    call check(o%partition     == SMC_COLUMN,       "default partition is SMC_COLUMN")
    call check(o%decompression == SMC_DIRECT,       "default decompression is SMC_DIRECT")
    call check(o%order         == SMC_NATURAL,      "default order is SMC_NATURAL")
    call check(o%postprocessing    == 0,            "default postprocessing is 0")
    call check(o%symmetric_pattern == 0,            "default symmetric_pattern is 0")
    call check(o%index_base        == 0,            "default index_base is 0")
    call check(o%dtype         == SMC_FLOAT64,      "default dtype is SMC_FLOAT64")
  end subroutine test_default_options

  ! A c_null_ptr options argument must behave exactly like smc_default_options().
  subroutine test_null_options()
    type(SmcColoringOptions), target :: o
    type(c_ptr) :: r_null, r_def
    integer(c_int), target :: a(MAXDIM), b(MAXDIM)
    integer(c_int) :: ret, nn
    integer :: j
    logical :: same

    write(*,'(A)') "c_null_ptr options ..."
    o = smc_default_options()
    nn = NONSYM(1)%n
    a = 0
    b = 0

    r_null = c_null_ptr
    r_def  = c_null_ptr
    ret = smc_coloring(NONSYM(1)%m, nn, c_loc(NONSYM(1)%colptr),               &
                       c_loc(NONSYM(1)%rowval), c_null_ptr, r_null)
    call check(ret == 0, "smc_coloring accepts c_null_ptr options")
    ret = color_matrix(NONSYM(1), o, r_def)
    call check(ret == 0, "smc_coloring accepts smc_default_options()")

    ret = smc_column_colors(r_null, c_loc(a), nn)
    call check(ret == 0, "colors with c_null_ptr options")
    ret = smc_column_colors(r_def, c_loc(b), nn)
    call check(ret == 0, "colors with explicit defaults")

    same = .true.
    do j = 1, nn
      if (a(j) /= b(j)) same = .false.
    end do
    call check(same, "c_null_ptr options == smc_default_options()")

    ret = smc_result_free(r_null)
    call check(ret == 0, "free")
    ret = smc_result_free(r_def)
    call check(ret == 0, "free")
  end subroutine test_null_options

  ! Return code -2: an unsupported (structure, partition, decompression).
  subroutine test_unsupported_combinations()
    integer :: c, dt
    type(SmcColoringOptions), target :: o
    type(c_ptr) :: result
    integer(c_int), target :: rows(MAXDIM), cols(MAXDIM), nc
    integer(c_int) :: ret

    write(*,'(A)') "unsupported combinations ..."
    do c = 1, 6
      do dt = 0, 1
        o = smc_default_options()
        o%structure     = UNSUPPORTED(1,c)
        o%partition     = UNSUPPORTED(2,c)
        o%decompression = UNSUPPORTED(3,c)
        o%dtype         = int(dt, c_int)

        result = c_null_ptr
        ret = color_matrix(SYMM(1), o, result)
        call check(ret == -2, "an unsupported combination returns -2")

        nc = -1
        rows = 0
        cols = 0
        ret = smc_fast_coloring(SYMM(1)%m, SYMM(1)%n, c_loc(SYMM(1)%colptr),   &
                                c_loc(SYMM(1)%rowval), c_loc(o),               &
                                c_loc(rows), c_loc(cols), c_loc(nc))
        call check(ret == -2, "smc_fast_coloring rejects the same combination with -2")
      end do
    end do
  end subroutine test_unsupported_combinations

  ! Return code -3: invalid arguments and short buffers.
  subroutine test_invalid_arguments()
    type(SmcColoringOptions), target :: o, bad
    type(c_ptr) :: result
    integer(c_int), target :: colors(MAXDIM), members(MAXDIM)
    integer(c_int), target :: ngroups, gsize
    integer(c_int) :: ret, mm, nn
    integer :: f

    write(*,'(A)') "invalid arguments ..."
    o = smc_default_options()
    mm = NONSYM(1)%m
    nn = NONSYM(1)%n
    result = c_null_ptr

    ret = smc_coloring(mm, nn, c_null_ptr, c_loc(NONSYM(1)%rowval), c_loc(o), result)
    call check(ret == -3, "a c_null_ptr colptr returns -3")
    ret = smc_coloring(mm, nn, c_loc(NONSYM(1)%colptr), c_null_ptr, c_loc(o), result)
    call check(ret == -3, "a c_null_ptr rowval returns -3")
    ret = smc_coloring(0_c_int, nn, c_loc(NONSYM(1)%colptr),                   &
                       c_loc(NONSYM(1)%rowval), c_loc(o), result)
    call check(ret == -3, "m == 0 returns -3")
    ret = smc_coloring(mm, 0_c_int, c_loc(NONSYM(1)%colptr),                   &
                       c_loc(NONSYM(1)%rowval), c_loc(o), result)
    call check(ret == -3, "n == 0 returns -3")
    ret = smc_coloring(-1_c_int, nn, c_loc(NONSYM(1)%colptr),                  &
                       c_loc(NONSYM(1)%rowval), c_loc(o), result)
    call check(ret == -3, "m < 0 returns -3")

    do f = 1, 6
      bad = smc_default_options()
      select case (f)
      case (1)
        bad%structure = 5
      case (2)
        bad%partition = 9
      case (3)
        bad%decompression = 7
      case (4)
        bad%order = 9
      case (5)
        bad%dtype = 4
      case default
        bad%index_base = 2
      end select
      result = c_null_ptr
      ret = color_matrix(NONSYM(1), bad, result)
      call check(ret == -3, "an out-of-range enum or index_base returns -3")
    end do

    result = c_null_ptr
    ret = color_matrix(NONSYM(1), o, result)
    call check(ret == 0, "reference coloring")

    ret = smc_column_colors(result, c_loc(colors), nn - 1)
    call check(ret == -3, "len < n returns -3")
    ret = smc_column_colors(result, c_null_ptr, nn)
    call check(ret == -3, "a c_null_ptr colors buffer returns -3")
    ret = smc_ncolors(result, c_null_ptr)
    call check(ret == -3, "a c_null_ptr ncolors_out returns -3")
    ret = smc_ncolumn_groups(result, c_null_ptr)
    call check(ret == -3, "a c_null_ptr ngroups_out returns -3")
    ret = smc_compressed_size(result, c_null_ptr, c_null_ptr, c_null_ptr, c_null_ptr)
    call check(ret == -3, "c_null_ptr size outputs return -3")
    ret = smc_nnz(result, c_null_ptr)
    call check(ret == -3, "a c_null_ptr nnz_out returns -3")
    ret = smc_size(result, c_null_ptr, c_null_ptr)
    call check(ret == -3, "a c_null_ptr m_out / n_out returns -3")

    ngroups = 0
    ret = smc_ncolumn_groups(result, c_loc(ngroups))
    call check(ret == 0 .and. ngroups > 0, "smc_ncolumn_groups succeeds")
    gsize = 0
    ret = smc_column_group_size(result, 0_c_int, c_loc(gsize))
    call check(ret == -3, "group 0 is out of range")
    ret = smc_column_group_size(result, ngroups + 1, c_loc(gsize))
    call check(ret == -3, "group ngroups+1 is out of range")
    ret = smc_column_group_size(result, 1_c_int, c_loc(gsize))
    call check(ret == 0 .and. gsize > 0, "group 1 has a size")
    ! Only understate a length that is positive; see pitfall 2 in the header.
    if (gsize > 0) then
      ret = smc_column_group(result, 1_c_int, c_loc(members), gsize - 1)
      call check(ret == -3, "a short group buffer returns -3")
    end if
    ret = smc_column_group(result, 1_c_int, c_null_ptr, gsize)
    call check(ret == -3, "a c_null_ptr group buffer returns -3")

    ret = smc_row_colors(result, c_loc(colors), mm)
    call check(ret == -2, "smc_row_colors on a column partition returns -2")
    ret = smc_nrow_groups(result, c_loc(ngroups))
    call check(ret == -2, "smc_nrow_groups on a column partition returns -2")

    ret = smc_result_free(result)
    call check(ret == 0, "free")
  end subroutine test_invalid_arguments

  ! Return code -4: a freed or never-allocated handle.
  subroutine test_invalid_handle()
    type(SmcColoringOptions), target :: o
    type(c_ptr) :: result, bogus
    integer(c_int), target :: colors(MAXDIM), members(MAXDIM), value
    integer(c_int), target :: not_a_handle
    real(c_double), target :: buf(MAXDIM*MAXDIM)
    integer(c_int) :: ret, mm, nn
    integer(c_size_t) :: buf_len

    write(*,'(A)') "invalid and already-freed handles ..."
    o = smc_default_options()
    mm = NONSYM(1)%m
    nn = NONSYM(1)%n
    colors = 0
    members = 0
    value = 0
    buf = 0.0_c_double
    buf_len = int(mm, c_size_t) * int(nn, c_size_t)
    not_a_handle = 0

    result = c_null_ptr
    ret = color_matrix(NONSYM(1), o, result)
    call check(ret == 0, "coloring for the free test")
    ret = smc_result_free(result)
    call check(ret == 0, "the first free returns 0")
    ret = smc_result_free(result)
    call check(ret == -4, "a double free returns -4")

    ! Every entry point must reject the stale handle rather than dereference it.
    ret = smc_ncolors(result, c_loc(value))
    call check(ret == -4, "smc_ncolors after free returns -4")
    ret = smc_column_colors(result, c_loc(colors), nn)
    call check(ret == -4, "smc_column_colors after free returns -4")
    ret = smc_row_colors(result, c_loc(colors), mm)
    call check(ret == -4, "smc_row_colors after free returns -4")
    ret = smc_ncolumn_groups(result, c_loc(value))
    call check(ret == -4, "smc_ncolumn_groups after free returns -4")
    ret = smc_nrow_groups(result, c_loc(value))
    call check(ret == -4, "smc_nrow_groups after free returns -4")
    ret = smc_column_group_size(result, 1_c_int, c_loc(value))
    call check(ret == -4, "smc_column_group_size after free returns -4")
    ret = smc_column_group(result, 1_c_int, c_loc(members), nn)
    call check(ret == -4, "smc_column_group after free returns -4")
    ret = smc_row_group_size(result, 1_c_int, c_loc(value))
    call check(ret == -4, "smc_row_group_size after free returns -4")
    ret = smc_row_group(result, 1_c_int, c_loc(members), mm)
    call check(ret == -4, "smc_row_group after free returns -4")
    ret = smc_compressed_size(result, c_loc(value), c_loc(value), c_loc(value), c_loc(value))
    call check(ret == -4, "smc_compressed_size after free returns -4")
    ret = smc_nnz(result, c_loc(value))
    call check(ret == -4, "smc_nnz after free returns -4")
    ret = smc_size(result, c_loc(value), c_loc(value))
    call check(ret == -4, "smc_size after free returns -4")
    ret = smc_compress(result, c_loc(NONSYM(1)%nzval), int(NONSYM(1)%nnz, c_size_t), &
                       c_null_ptr, 0_c_size_t, c_loc(buf), buf_len)
    call check(ret == -4, "smc_compress after free returns -4")
    ret = smc_decompress(result, c_null_ptr, 0_c_size_t, c_loc(buf), buf_len,  &
                         c_loc(buf), buf_len)
    call check(ret == -4, "smc_decompress after free returns -4")

    bogus = c_loc(not_a_handle)
    ret = smc_result_free(bogus)
    call check(ret == -4, "freeing a never-allocated handle returns -4")
    ret = smc_ncolors(bogus, c_loc(value))
    call check(ret == -4, "querying a never-allocated handle returns -4")

    ret = smc_result_free(c_null_ptr)
    call check(ret == -3 .or. ret == -4, "freeing c_null_ptr is rejected, not a crash")
  end subroutine test_invalid_handle

  subroutine test_sizing_queries()
    integer :: c, dt
    type(SmcColoringOptions), target :: o
    type(c_ptr) :: result
    integer(c_int), target :: got_nnz, got_m, got_n
    integer(c_int) :: ret, want_m, want_n, want_nnz
    logical :: symmetric

    write(*,'(A)') "smc_nnz / smc_size ..."
    do c = 1, 6
      do dt = 0, 1
        o = smc_default_options()
        o%structure     = SUPPORTED(1,c)
        o%partition     = SUPPORTED(2,c)
        o%decompression = SUPPORTED(3,c)
        o%dtype         = int(dt, c_int)

        symmetric = (SUPPORTED(1,c) == SMC_SYMMETRIC)
        if (symmetric) then
          want_m   = SYMM(1)%m
          want_n   = SYMM(1)%n
          want_nnz = SYMM(1)%nnz
        else
          want_m   = NONSYM(1)%m
          want_n   = NONSYM(1)%n
          want_nnz = NONSYM(1)%nnz
        end if

        result = c_null_ptr
        if (symmetric) then
          ret = color_matrix(SYMM(1), o, result)
        else
          ret = color_matrix(NONSYM(1), o, result)
        end if
        call check(ret == 0, "coloring for the sizing queries")

        got_nnz = -1
        ret = smc_nnz(result, c_loc(got_nnz))
        call check(ret == 0 .and. got_nnz == want_nnz,                         &
                   "smc_nnz is the number of stored entries")

        got_m = -1
        got_n = -1
        ret = smc_size(result, c_loc(got_m), c_loc(got_n))
        call check(ret == 0 .and. got_m == want_m .and. got_n == want_n,       &
                   "smc_size is the shape of the colored matrix")

        ret = smc_nnz(result, c_null_ptr)
        call check(ret == -3, "a c_null_ptr nnz_out returns -3")
        ret = smc_size(result, c_null_ptr, c_loc(got_n))
        call check(ret == -3, "a c_null_ptr m_out returns -3")
        ret = smc_size(result, c_loc(got_m), c_null_ptr)
        call check(ret == -3, "a c_null_ptr n_out returns -3")

        ret = smc_result_free(result)
        call check(ret == 0, "free")
        ret = smc_nnz(result, c_loc(got_nnz))
        call check(ret == -4, "smc_nnz on a freed handle returns -4")
        ret = smc_size(result, c_loc(got_m), c_loc(got_n))
        call check(ret == -4, "smc_size on a freed handle returns -4")
      end do
    end do
  end subroutine test_sizing_queries

end program test_smc
